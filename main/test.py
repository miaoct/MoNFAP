'''
Function:
    Implementation of Tester
Author:
    Zhenchao Jin
'''
import sys
sys.path.append('')
import os
import copy
import torch
import time
import warnings
import pickle
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from datasets import BuildDataset
from models import BuildSegmentor
from utils import (
    BuildDistributedDataloader, BuildDistributedModel, touchdir, loadckpts, postprocesspredgtpairs, 
    Logger, ConfigParser, ImgMetric, PixMetric, setrandomseed,
)
warnings.filterwarnings('ignore')


'''Tester'''
class Tester():
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
        self.pix_metric = PixMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
        # output path
        self.output_path = os.path.join(cfg.SEGMENTOR_CFG['output_dir'], cfg.SEGMENTOR_CFG['work_dir'])
    '''start tester'''
    def start(self, all_segpreds, all_imgpreds, all_imggts):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        rank_id = cmd_args.local_rank
        # build dataset and dataloader
        cfg.SEGMENTOR_CFG['dataset']['evalmode'] = self.cmd_args.evalmode
        dataset = BuildDataset(mode='test', logger_handle=logger_handle, dataset_cfg=cfg.SEGMENTOR_CFG['dataset'])
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.SEGMENTOR_CFG['dataloader'])
        dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers'] = dataloader_cfg['test'].pop('batch_size_per_gpu'), dataloader_cfg['test'].pop('num_workers_per_gpu')
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='test')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        # load ckpts
        if cmd_args.evalmode == 'online':
            checkpointspath = os.path.join(self.output_path, 'epoch_best.pth')
        elif cmd_args.evalmode == 'offline':
            checkpointspath = cmd_args.ckptspath
        else:
            logger_handle.info('testing mode error! plaese select online or offline')
        ckpts = loadckpts(checkpointspath)
        try:
            segmentor.load_state_dict(ckpts['model'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load ckpts by using strict=False')
            segmentor.load_state_dict(ckpts['model'], strict=False)
        # parallel
        segmentor = BuildDistributedModel(segmentor, {'device_ids': [cmd_args.local_rank]})
        # print information
        if cmd_args.local_rank == 0:
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'Config details: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.ckptspath}')
        # set eval
        segmentor.eval()
        # start to test
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


'''parse arguments in command line'''
def parsecmdargs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', help='node rank for distributed testing', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--evalmode', dest='evalmode', help='evaluate mode, support server and local', default='online', type=str, choices=['online', 'offline'])
    parser.add_argument('--ckptspath', dest='ckptspath', help='checkpoints you want to resume from', type=str)
    parser.add_argument('--random_seed', dest='random_seed', help='set random seed', default=666, type=int)
    args = parser.parse_args()
    if torch.__version__.startswith('2.'):
        args.local_rank = int(os.environ['LOCAL_RANK'])
    return args


'''main'''
def main():
    # parse arguments
    args = parsecmdargs()
    cfg, cfg_file_path = ConfigParser()(args.cfgfilepath)
    # seed
    setrandomseed(args.random_seed)
    # touch work dir
    output_path = os.path.join(cfg.SEGMENTOR_CFG['output_dir'], cfg.SEGMENTOR_CFG['work_dir'])
    touchdir(output_path)
    # initialize logger_handle
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_test.log".format(time_str, cfg.SEGMENTOR_CFG['logfilepath'])
    logger_handle = Logger(os.path.join(output_path, log_name))
    # number of gpus, for distribued testing, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != args.nproc_per_node:
        if args.local_rank == 0:
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Tester
    all_segpreds, all_imgpreds, all_imggts = [], [], []
    client = Tester(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start(all_segpreds, all_imgpreds, all_imggts)
    img_metric = ImgMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
    pix_metric = PixMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
    # post process
    rank_id = args.local_rank
    filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('/')[-1].split('.')[0] + f'_{rank_id}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
    with open(os.path.join(output_path, filename), 'wb') as fp:
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
            fp = open(os.path.join(output_path, filename), 'rb')
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
        # fake sample pixel-level
        fake_total_matrix_filtered, fake_all_ids = np.zeros((cfg.SEGMENTOR_CFG['num_classes'], cfg.SEGMENTOR_CFG['num_classes'])), []
        for idx, pred in enumerate(all_segpreds_gather):
            if pred[0] in fake_all_ids: 
                continue
            fake_all_ids.append(pred[0])
            if all_imggts_gather[idx] == 1:
                fake_total_matrix_filtered += pred[1]
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
        img_metric.update(all_imgpreds_filtered, all_imggts_filtered)
        img_acc, img_auc, img_f1, img_eer = img_metric.total_score()
        logger_handle.info("img_acc: {:.4f} || img_auc: {:.4f} || img_f1: {:.4f} || img_eer: {:.4f}".format(
                            img_acc, img_auc, img_f1, img_eer))
        
        f1_fake, f1_real, mIoU_fake, mIoU_real =pix_metric.total_score(total_matrix_filtered)
        logger_handle.info("pix_tol_f1: {:.4f} || pix_tol_mIoU: {:.4f} || pix_fake_f1: {:.4f} || pix_fake_mIoU: {:.4f}".format(
                            (f1_fake+f1_real)/2, (mIoU_fake+mIoU_real)/2, f1_fake, mIoU_fake))
        
        fake_f1_fake, fake_f1_real, fake_mIoU_fake, fake_mIoU_real =pix_metric.total_score(fake_total_matrix_filtered)
        logger_handle.info("fake_pix_tol_f1: {:.4f} || fake_pix_tol_mIoU: {:.4f} || fake_pix_fake_f1: {:.4f} || fake_pix_fake_mIoU: {:.4f}".format(
                            (fake_f1_fake+fake_f1_real)/2, (fake_mIoU_fake+fake_mIoU_real)/2, fake_f1_fake, fake_mIoU_fake))
        # remove .pkl file
        for rank in rank_list:
            rank = str(int(rank.item()))
            filename = cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[0] + f'_{rank}.' + cfg.SEGMENTOR_CFG['resultsavepath'].split('.')[-1]
            if os.path.exists(os.path.join(output_path, filename)):
                os.remove(os.path.join(output_path, filename))
                print('Remove result.pkl')


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        main()