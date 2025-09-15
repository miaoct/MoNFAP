'''
Function:
    Implementation of Trainer
Author:
    Changtao MIao
    refer to Zhenchao Jin 'https://github.com/SegmentationBLWX/sssegmentation'
'''
import sys
sys.path.append('')
import os
import time
import copy
import torch
import pickle
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from datasets import BuildDataset
from models import BuildSegmentor
from utils import (
    BuildDistributedDataloader, BuildDistributedModel, touchdir, loadckpts, saveckpts, judgefileexist, postprocesspredgtpairs, 
    BuildOptimizer, BuildScheduler, Logger, ConfigParser, ImgMetric, PixMetric, setrandomseed,
)
warnings.filterwarnings('ignore')


'''Trainer'''
class Trainer():
    def __init__(self, cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path):
        # set attribute
        self.cfg = cfg
        self.ngpus_per_node = ngpus_per_node
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
         # init metric
        self.img_metric = ImgMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
        self.pix_metric = PixMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
        self.best_pred = 0.0
        # output path
        self.output_path = os.path.join(cfg.SEGMENTOR_CFG['output_dir'], cfg.SEGMENTOR_CFG['work_dir'])
    '''start trainer'''
    def start(self):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # build dataset and dataloader
        dataset = BuildDataset(mode='train', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers'] = dataloader_cfg['train'].pop('batch_size_per_gpu'), dataloader_cfg['train'].pop('num_workers_per_gpu')
        dataloader_cfg['val']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['val'].pop('batch_size_per_gpu'), dataloader_cfg['val'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'])
        # build segmentor
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='train')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer = BuildOptimizer(segmentor, cfg.SEGMENTOR_CFG['scheduler']['optimizer'])
        # build scheduler
        scheduler_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['scheduler'])
        scheduler_cfg.update({
            'lr': cfg.SEGMENTOR_CFG['scheduler']['optimizer']['lr'],
            'iters_per_epoch': len(dataloader),
            'params_rules': cfg.SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'],
        })
        scheduler = BuildScheduler(optimizer=optimizer, scheduler_cfg=scheduler_cfg)
        start_epoch, end_epoch = 1, scheduler_cfg['max_epochs']
        # load ckpts
        if cmd_args.ckptspath and judgefileexist(cmd_args.ckptspath):
            ckpts = loadckpts(cmd_args.ckptspath)
            try:
                segmentor.load_state_dict(ckpts['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False')
                segmentor.load_state_dict(ckpts['model'], strict=False)
            if 'optimizer' in ckpts: 
                optimizer.load_state_dict(ckpts['optimizer'])
            if 'cur_epoch' in ckpts: 
                start_epoch = ckpts['cur_epoch'] + 1
                scheduler.setstate({'cur_epoch': ckpts['cur_epoch'], 'cur_iter': ckpts['cur_iter']})
                assert ckpts['cur_iter'] == len(dataloader) * ckpts['cur_epoch']
        else:
            cmd_args.ckptspath = ''
        # parallel segmentor
        build_dist_model_cfg = self.cfg.SEGMENTOR_CFG.get('build_dist_model_cfg', {})
        build_dist_model_cfg.update({'device_ids': [cmd_args.local_rank]})
        segmentor = BuildDistributedModel(segmentor, build_dist_model_cfg)
        # print config
        if cmd_args.local_rank == 0:
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.ckptspath}')
        # start to train the segmentor
        FloatTensor, losses_log_dict_memory = torch.cuda.FloatTensor, {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --train epoch
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in enumerate(dataloader):
                pbar.set_description('Training %s/%s in rank %s' % (batch_idx+1, len(dataloader), cmd_args.local_rank))
                learning_rate = scheduler.updatelr()
                images = samples_meta['image'].type(FloatTensor)
                targets = {'seg_target': samples_meta['seg_target'].type(FloatTensor), 'label': samples_meta['img_target'].type(FloatTensor)}
                optimizer.zero_grad()
                loss, losses_log_dict = segmentor(images, targets)
                for key, value in losses_log_dict.items():
                    if key in losses_log_dict_memory: 
                        losses_log_dict_memory[key].append(value)
                    else: 
                        losses_log_dict_memory[key] = [value]
                loss.backward()
                for name, param in segmentor.named_parameters():
                    if param.grad is None:
                        print(name)
                scheduler.step()
                if (cmd_args.local_rank == 0) and (scheduler.cur_iter % cfg.SEGMENTOR_CFG['log_interval_iterations'] == 0):
                    print_log = {
                        'cur_epoch': epoch, 'max_epochs': end_epoch, 'cur_iter': scheduler.cur_iter, 'max_iters': scheduler.max_iters,
                        'segmentor': cfg.SEGMENTOR_CFG['type'], 
                        'backbone': cfg.SEGMENTOR_CFG['backbone']['structure_type'], 'dataset': cfg.SEGMENTOR_CFG['dataset']['type'], 
                        'learning_rate': learning_rate,
                    }
                    for key in list(losses_log_dict_memory.keys()):
                        print_log[key] = sum(losses_log_dict_memory[key]) / len(losses_log_dict_memory[key])
                    logger_handle.info(print_log)
                    losses_log_dict_memory = dict()
                break
            scheduler.cur_epoch = epoch
            # --save ckpts
            if (epoch % cfg.SEGMENTOR_CFG['save_interval_epochs'] == 0) or (epoch == end_epoch):
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                savepath = os.path.join(self.output_path, 'epoch_%s.pth' % epoch)
                if cmd_args.local_rank == 0:
                    saveckpts(state_dict, savepath)
            # remove before checkpoints
            if os.path.exists(os.path.join(self.output_path, 'epoch_%s.pth' % (epoch-2))) and cmd_args.local_rank == 0:
                    os.remove(os.path.join(self.output_path, 'epoch_%s.pth' % (epoch-2)))
                    print('Remove epoch_%s.pth' % (epoch-2))
            # --eval ckpts
            if (epoch % cfg.SEGMENTOR_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                if cmd_args.local_rank == 0: 
                    self.logger_handle.info(f'Evaluate {cfg.SEGMENTOR_CFG["type"]} at epoch {epoch}')
                self.evaluate(segmentor, scheduler)
    '''evaluate'''
    def evaluate(self, segmentor, scheduler):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # TODO: bug occurs if use --pyt bash
        rank_id = cmd_args.local_rank
        # build dataset and dataloader
        dataset = BuildDataset(mode='val', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['val']['batch_size'], dataloader_cfg['val']['num_workers'] = dataloader_cfg['val'].pop('batch_size_per_gpu'), dataloader_cfg['val'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['val'])
        # start to eval
        self.img_metric.reset()
        # self.pix_metric.reset()
        segmentor.eval()
        segmentor.module.mode = 'val'
        all_segpreds, all_imgpreds, all_imggts = [], [], []
        align_corners = segmentor.module.align_corners
        FloatTensor = torch.cuda.FloatTensor
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples_meta in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, seggts, imggts = samples_meta['id'], samples_meta['image'].type(FloatTensor), samples_meta['width'], samples_meta['height'], samples_meta['seg_target'], samples_meta['img_target']
                img_outputs, pix_outputs = segmentor.module.inference(images)
                # pix-level
                pix_outputs = F.interpolate(pix_outputs, size=(heights, widths), mode='bilinear', align_corners=align_corners)
                pix_pred = (torch.argmax(pix_outputs, dim=1)).cpu().numpy().astype(np.int32)
                seggt = seggts.cpu().numpy().astype(np.int32)
                seggt[seggt >= dataset.num_classes] = -1
                confusionMatrix = self.pix_metric.BatchConfusionMatrix(pix_pred, seggt)
                all_segpreds.append([imageids, confusionMatrix])
                # image-level
                img_pred = F.softmax(img_outputs, dim=1).data.cpu().numpy()[:, 1].astype(np.float64)
                all_imgpreds.append([imageids, img_pred])
                imggt = imggts.cpu().numpy().astype(np.float64)
                all_imggts.append(imggt)
                
        # post process
        # collect eval results and calculate the metric
        filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank_id}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
        with open(os.path.join(self.output_path, filename), 'wb') as fp:
            pickle.dump([all_segpreds, all_imgpreds, all_imggts], fp)
        rank = torch.tensor([rank_id], device='cuda')
        rank_list = [rank.clone() for _ in range(ngpus_per_node)]
        dist.all_gather(rank_list, rank)
        logger_handle.info('Rank %s finished' % int(rank.item()))
        if rank_id == 0:
            all_segpreds_gather, all_imgpreds_gather, all_imggts_gather = [], [], []
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[0] + f'_{rank}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
                fp = open(os.path.join(self.output_path, filename), 'rb')
                all_segpreds, all_imgpreds, all_imggts = pickle.load(fp)
                all_segpreds_gather += all_segpreds
                all_imgpreds_gather += all_imgpreds
                all_imggts_gather += all_imggts
            # pixel-level
            total_matrix_filtered, all_ids = np.zeros((cfg.SEGMENTOR_CFG['num_classes'], cfg.SEGMENTOR_CFG['num_classes'])), []
            for pred in all_segpreds_gather:
                if pred[0] in all_ids: 
                    continue
                all_ids.append(pred[0])
                total_matrix_filtered += pred[1]
            # image-level
            all_imgpreds_filtered, all_imggts_filtered, all_ids = [], [], []
            for idx, pred in enumerate(all_imgpreds_gather):
                if pred[0] in all_ids: 
                    continue
                all_ids.append(pred[0])
                all_imgpreds_filtered.append(pred[1])
                all_imggts_filtered.append(all_imggts_gather[idx])
            # calculate the metric
            logger_handle.info('All Finished, all_preds: %s' % (len(all_imgpreds_filtered)))
            self.img_metric.update(all_imgpreds_filtered, all_imggts_filtered)
            img_acc, img_auc, img_f1, img_eer = self.img_metric.total_score()
            logger_handle.info("img_acc: {:.4f} || img_auc: {:.4f} || img_f1: {:.4f} || img_eer: {:.4f}".format(
                                img_acc, img_auc, img_f1, img_eer))
            
            f1_fake, f1_real, mIoU_fake, mIoU_real =self.pix_metric.total_score(total_matrix_filtered)
            logger_handle.info("pix_tol_f1: {:.4f} || pix_tol_mIoU: {:.4f} || pix_fake_f1: {:.4f} || pix_fake_mIoU: {:.4f}".format(
                                (f1_fake+f1_real)/2, (mIoU_fake+mIoU_real)/2, f1_fake, mIoU_fake))
            # remove .pkl file
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[0] + f'_{rank}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
                if os.path.exists(os.path.join(self.output_path, filename)):
                    os.remove(os.path.join(self.output_path, filename))
                    print('Remove result.pkl')
            # --save best checkpoints
            tol_balance_pred = mIoU_fake
            if tol_balance_pred > self.best_pred:
                self.best_pred = tol_balance_pred
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                savepath = os.path.join(self.output_path, 'epoch_best.pth')
                saveckpts(state_dict, savepath)
                
        segmentor.train()
        segmentor.module.mode = 'train'


'''parse arguments in command line'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', default='', type=str)
    parser.add_argument('--random_seed', dest='random_seed', help='set random seed', default=866, type=int)
    cmd_args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        cmd_args.local_rank = int(os.environ['LOCAL_RANK'])
    return cmd_args


'''main'''
def main():
    # parse arguments
    cmd_args, config_parser = parsecmdargs(), ConfigParser()
    cfg, cfg_file_path = config_parser(cmd_args.cfgfilepath)
    # seed
    setrandomseed(cmd_args.random_seed)
    # touch work dir
    output_path = os.path.join(cfg.SEGMENTOR_CFG['output_dir'], cfg.SEGMENTOR_CFG['work_dir'])
    touchdir(output_path)
    config_parser.save(output_path)
    # initialize logger_handle
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}.log".format(time_str, cfg.SEGMENTOR_CFG['logfilepath'])
    logger_handle = Logger(os.path.join(output_path, log_name))
    # number of gpus, for distribued training, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != cmd_args.nproc_per_node:
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0): 
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = cmd_args.nproc_per_node
    # instanced Trainer
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=cmd_args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()