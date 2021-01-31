import jittor as jt
#import torch.backends.cudnn as cudnn #没找到替代的
from jittor import init
import jittor.nn as nn
from jittor.dataset import Dataset  #不知道行不行
from jittor.dataset import ImageFolder
from jittor import transform
import jittor.models as jtmodels
import torchvision.models as tcmodels   #用于加载pytorch下的vgg参数到jittor下的vgg

from jittor.dataset.utils import get_random_list, get_order_list, collate_batch, HookTimer
from collections.abc import Sequence, Mapping

from PIL import Image
from PIL import ImageFile
import argparse
import os

from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import torch

import numpy as np
import JTnet as net

#from JTsampler import InfiniteSamplerWrapper

#cudnn.benchmark = True   #自动适配最高效算法  #jittor中未找到匹配参数

jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU

Image.MAX_IMAGE_PIXELS = None   #无最大图像限制检查
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():   #图像预处理方式
    transform_list = [
        transform.Resize(size=(512,512)),
        transform.RandomCrop(256),
        transform.ToTensor()
    ]
    return transform.Compose(transform_list)

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform, train=True):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        #print(self.root)
        self.path = os.listdir(self.root)   #os.listdir可以返回指定文件下的所有内容   将文件夹下所有内容保存在path变量中
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)
                    #print(self.root + "/" + file_name+ "/" + file_name1)

        else:
            self.paths = list(Path(self.root).glob('*'))
        #print(self.paths)
        self.transform = transform

        self.len = len(self.paths)

    def __getitem__(self,index):  #getitem方法，当调用类名[index]时会调用这个函数  返回第i个image
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

    def __iter__(self):
        i = self.len - 1
        batch_size = args.batch_size
        index_list = get_random_list(self.len)
        batch_data = []
        while True:
            for x in range(batch_size):
                y = i
                if i >= self.len:
                    y = i - self.len
                batch_data.append(self[index_list[y]])
                i += 1

            if (i >= self.len):
                #jt.set_seed()
                index_list = get_random_list(self.len)
                i = 0

            batch_data = self.collate_batch(batch_data)
            batch_data = self.to_jittor(batch_data)
            yield jt.float(batch_data)
            batch_data = []


def adjust_learning_rate(optimizer, iteration_count):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()  #命令行接口

# Basic options
parser.add_argument('--content_dir', default='../../datasets/train2014', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='../../datasets/Images', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')

#training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr' ,type=float,default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=3.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

writer = SummaryWriter(log_dir = args.log_dir)

decoder = net.decoder

#加载vgg参数
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))


vgg = nn.Sequential(*list(vgg.children())[:44])
with jt.no_grad():
    network = net.Net(vgg, decoder)
network.train()

#decoder.eval()  #decoder设置不用学习
#network.to(device)  #是指让这个网络在指定device上运行吗
#network = nn.DataParallel(network, device_ids=[0,1])  ##这句也没找到合适的

content_tf = train_transform()
style_tf = train_transform()


#jittor这里直接对获取的数据集进行lorader，不分开成功两步
#content_dataset = FlatFolderDataset(args.content_dir, content_tf)
#style_dataset = FlatFolderDataset(args.style_dir, style_tf)  #获得两个数据集，并预处理 #数据集只是一个数组存放了文件  #在抽取一组batch时进行预处理

#content_iter = iter(Dataset(    #生成迭代器... 将dataloader里的数据生成迭代器，类似enumerate..?
#    style_dataset, batch_size=args.batch_size,
#    sampler=InfiniteSamplerWrapper(style_dataset),
#    num_worker = args.n_thread)
#)
#style_iter = iter(Dataset(
#    style_dataset, batch_size=args.batch_size,
#    sampler=InfiniteSamplerWrapper(style_dataset),
#    num_workers=args.n_threads))

content_dataset_loader = FlatFolderDataset(args.content_dir, content_tf).set_attrs(batch_size = args.batch_size, num_workers = args.n_threads)
style_dataset_loader = FlatFolderDataset(args.style_dir, content_tf).set_attrs(batch_size = args.batch_size, num_workers = args.n_threads)  #但是没有sampler参数

content_iter = iter(content_dataset_loader)
style_iter = iter(style_dataset_loader)



optimizer = jt.optim.Adam([
    {'params':network.decoder.parameters()},
    {'params':network.mcc_module.parameters()}],lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter)
    style_images = next(style_iter)

    loss_n, loss_c, loss_s, l_identity1, l_identity2, loss_tv = network(content_images, style_images)

#    print("loss_n:",loss_n)
#    print("loss_c:",loss_c)
#    print("loss_s:",loss_s)
#    print("l_identity1:",l_identity1)
#    print("l_identity2:",l_identity2)
#    print("loss_tv:", loss_tv)

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_tv = 10e-5 * loss_tv
    loss = 3000 * loss_n + loss_c + loss_s + (l_identity1 *70 ) + (l_identity2 * 1) + loss_tv



#    print(loss.sum().cpu().detach().numpy(), "-content:", loss_s.sum().cpu().detach().numpy(), "-style:", loss_s.sum().cpu().detach().numpy()
#          ,"-l1:",l_identity1.sum().cpu().detach().numpy(), "-l2:", l_identity2.sum().cpu().detach().numpy(), loss_tv.sum().cpu().detach().numpy())
    print(loss.data, "-content:", loss_c.data, "-style:", loss_s.data,"-l1:",l_identity1.data, "-l2:", l_identity2.data, loss_tv.data)

    Ics = network.Icstest[0]
    writer.add_image('Ics',Ics.numpy(),i+1)

    #optimizer.zero_grad() #梯度置为0，不然会累加计算
    #loss().sum().sync()
    optimizer.step(loss.sum())

    writer.add_scalar('loss_content', loss_c.data, i + 1)
    writer.add_scalar('loss_style', loss_s.data, i + 1)
    writer.add_scalar('loss_identity1', l_identity1.data, i + 1)
    writer.add_scalar('loss_identity2', l_identity2.data, i + 1)
    writer.add_scalar('total_loss', loss.data, i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        network.decoder.save('{:s}/decoder_iter_{:d}.pkl'.format(args.save_dir, i+1))
        #state_dict = network.decoder.state_dict()
        #jt.save(state_dict,
        #        '{:s}/decoder_iter_{:d}.pkl'.format(args.save_dir, i+1))  #周期性的保存权重

        network.mcc_module.save('{:s}/mcc_module_iter_{:d}.pkl'.format(args.save_dir, i+1 ))
        #state_dict = network.mcc_module.state_dict()
        #jt.save(state_dict,
        #        '{:s}/mcc_module_iter_{:d}.pkl'.format(args.save_dir, i+1 ))

    #jt.sync_all()
#jt.sync_all(device_sync=True)
writer.close()



