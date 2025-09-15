'''
Function:
    Implementation of BuildDistributedDataloader
Author:
    Zhenchao Jin
'''
import math
import random
import copy
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

'''BuildDistributedDataloader'''
def BuildDistributedDataloader(dataset, dataloader_cfg):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    # build dataloader
    shuffle = dataloader_cfg.pop('shuffle')
    dataloader_cfg['shuffle'] = False
    dataloader_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    # return
    return dataloader

def BuildDistributedDataloaderTrain(dataset, dataloader_cfg):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    # build dataloader
    shuffle = dataloader_cfg.pop('shuffle')
    dataloader_cfg['shuffle'] = False
    # dataloader_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader_cfg['sampler'] = BalancedDistributedSampler(dataset, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_cfg)
    # return
    return dataloader



# class BalancedDistributedSampler(DistributedSampler):
#     """
#     Balanced DistributedSampler that ensures each class is represented in each mini-batch.

#     Args:
#         dataset: Dataset to sample from.
#         num_replicas: Number of distributed processes.
#         rank: Rank of the current process.
#         weights: Dictionary mapping class labels to their respective weights.
#     """

#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, weights=):
#         super().__init__(dataset, num_replicas, rank)
#         self.weights = weights

#     def __iter__(self):
#         # Get the class labels for the dataset
#         class_labels = [sample["img_target"] for sample in self.dataset]

#         # Create a weighted sampler for each class
#         class_samplers = {}
#         for class_label in set(class_labels):
#             class_sampler = WeightedRandomSampler(class_labels, self.weights[class_label])
#             class_samplers[class_label] = class_sampler

#         # Sample data from each class
#         sampled_indices = []
#         for class_label in set(class_labels):
#             class_sampler = class_samplers[class_label]
#             sampled_indices.extend(class_sampler(len(self.dataset)))

#         # Shuffle the sampled indices
#         random.shuffle(sampled_indices)

#         # Return the sampled indices
#         return iter(sampled_indices)

# class BalancedDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

#         # Get the class labels from the dataset
#         self.labels = dataset.num_classes

#         # Compute the class distribution
#         class_counts = torch.bincount(torch.tensor(self.labels))
#         self.class_weights = 1.0 / class_counts.float()

#         # Broadcast the class weights to all processes
#         if self.num_replicas is not None:
#             self.class_weights = self.class_weights / dist.get_world_size()
#             dist.broadcast(self.class_weights, src=0)

#     def __iter__(self):
#         # Get the indices from the parent class
#         indices = super().__iter__()

#         # Sort the indices by class label
#         indices.sort(key=lambda i: self.labels[i])

#         return iter(indices)

class BalancedDistributedSampler(DistributedSampler):
    """
    自定义 DistributedSampler 实现批次类别平衡。

    Args:
        dataset: 数据集对象。
        num_replicas: 分布式训练中进程数量。
        rank: 当前进程的秩。
        shuffle: 是否对数据进行洗牌。
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # 统计每个类别的样本数量
        self.class_counts = {}
        for sample in self.dataset:
            label = sample["img_target"]
            if label not in self.class_counts:
                self.class_counts[label] = 0
            self.class_counts[label] += 1

        # 计算每个类别在每个批次中应有的样本数量
        self.num_samples_per_class = [int(count / num_replicas) for count in self.class_counts.values()]

        # 填充每个批次，确保每个类别都有足够的样本
        self.batch_indices = []
        for class_index in range(len(self.class_counts)):
            for i in range(self.num_samples_per_class[class_index]):
                self.batch_indices.append(class_index)

        # 洗牌批次索引（可选）
        if self.shuffle:
            random.shuffle(self.batch_indices)

    def __iter__(self):
        # 循环批次索引
        for batch_index in self.batch_indices:
            # 获取当前进程的样本索引
            sample_index = batch_index + self.rank * len(self.batch_indices) // self.num_replicas

            # 确保样本索引在数据集范围内
            sample_index = sample_index % len(self.dataset)

            yield sample_index

    def __len__(self):
        # 返回批次数量
        return len(self.batch_indices)


# # # 自己实现一个batchsampler 采样器，精准控制每个batch里得类别数量
# https://i.steer.space/blog/2020/12/pytorch-ddp-sampler-warper#fnref:1
# class MyBatchSampler(Sampler):
#     def __init__(self, data_source, batch_size, class_weight):
#         super(MyBatchSampler, self).__init__(data_source)
#         self.data_source = data_source
#         assert isinstance(class_weight, list)
#         assert 1 - sum(class_weight) < 1e-5
#         self.batch_size = batch_size

#         _num = len(class_weight)
#         number_in_batch = {i: 0 for i in range(_num)}
#         for c in range(_num):
#             number_in_batch[c] = math.floor(batch_size * class_weight[c])
#         _remain_num = batch_size - sum(number_in_batch.values())
#         number_in_batch[random.choice(range(_num))] += _remain_num
#         self.number_in_batch = number_in_batch
#         self.offset_per_class = {i: 0 for i in range(_num)}
#         print(f'setting number_in_batch: {number_in_batch}')
#         print('my sampler is inited.')

#         # 如果是分布式，需要重新分配采样比例，避免重复采样
#         if dist.is_available() and dist.is_initialized():
#             rank = dist.get_rank()
#             num_replicas = dist.get_world_size()
#             t = self.data_source.class_idx_samples.items()
#             for c, (start, end) in t:
#                 total = end - start
#                 num_samples = math.ceil(total / num_replicas)
#                 start_rank = rank * num_samples + start
#                 end_rank = start_rank + num_samples
#                 if end_rank > end: end_rank = end
#                 # update idx range
#                 self.data_source.class_idx_samples[c] = [start_rank, end_rank]

#             print('using torch distributed mode.')
#             print(f'current rank data sample setting: {self.data_source.class_idx_samples}')

#     def __iter__(self):
#         print('======= start __iter__ =======')
#         batch = []
#         i = 0
#         while i < len(self):
#             for c, num in self.number_in_batch.items():
#                 start, end = self.data_source.class_idx_samples[c]
#                 for _ in range(num):
#                     idx = start + self.offset_per_class[c]
#                     if idx >= end:
#                         self.offset_per_class[c] = 0
#                     idx = start + self.offset_per_class[c]
#                     batch.append(idx)
#                     self.offset_per_class[c] += 1

#             assert len(batch) == self.batch_size
#             # random.shuffle(batch)
#             yield batch
#             batch = []
#             i += 1

#     def __len__(self):
#         return len(self.data_source) // self.batch_size