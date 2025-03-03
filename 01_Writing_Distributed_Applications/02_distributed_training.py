"""
https://pytorch.org/tutorials/intermediate/dist_tuto.html
https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
https://sebarnold.net/dist_blog/
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
"""
#!/usr/bin/env python
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients_all_reduce(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    group = dist.new_group([0, 1])
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=group)
            param.grad.data /= size


def ring_all_reduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone()
    recv_buff = send.clone()
    accum = send.clone()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv_buff[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]


def average_gradients_ring_all_reduce(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.Tensor:
            ring_all_reduce(param.grad.data, param.grad.data)
            param.grad.data /= size


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        # Option 1:
        # model = Net()
        # model = model.cuda(rank)
        # Option 2:
        device = torch.device("cuda:{}".format(rank))
        model = Net().to(device)
    else:
        model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            if cuda_is_available:
                # Option 1:
                # data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
                # Option 2:
                data, target = data.to(device), target.to(device)
            else:
                data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            # Option 1:
            # average_gradients_all_reduce(model)
            # Option 2:
            average_gradients_ring_all_reduce(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    start_time = time.time()
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run, 'nccl'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end_time = time.time()
    print("elapsed time = {}".format(end_time-start_time))