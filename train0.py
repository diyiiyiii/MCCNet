import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net as  net
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


# class FlatFolderDataset(data.Dataset):
#     def __init__(self, root, transform):
#         super(FlatFolderDataset, self).__init__()
#         self.root = root
#         self.paths = os.listdir(self.root)
#         self.transform = transform

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.paths)

#     def name(self):
#         return 'FlatFolderDataset'

# class FlatFolderSDataset(data.Dataset):
#     def __init__(self, paths, transform):
#         super(FlatFolderSDataset, self).__init__()          
#         self.paths = paths 
        

#         #print(self.paths)
#         self.transform = transform

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(path).convert('RGB')
#         img = self.transform(img)
     
#         return img

#     def __len__(self):
#         return len(self.paths)

#     def name(self):
#         return 'FlatFolderDataset'
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)  
                    print(self.root+"/"+file_name+"/"+file_name1) 
            
            
        else:

            self.paths = list(Path(self.root).glob('*'))
        print(self.paths)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=3.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

decoder = net.decoder
#decoder = net.Decoder(net.decoder)
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
with torch.no_grad():
    network = net.Net(vgg, decoder)
network.train()
decoder.eval()
network.to(device)
network = nn.DataParallel(network, device_ids=[0,1])
content_tf = train_transform()
style_tf = train_transform()

# network.module.decoder.load_state_dict(torch.load("new_models/model-v09/decoder_iter_20000.pth"))
# network.module.sa_module.load_state_dict(torch.load("new_models/model-v09/sa_module_iter_20000.pth"))

content_dataset = FlatFolderDataset(args.content_dir, content_tf)

# paths = []
# for i in os.listdir(args.style_dir):
#     for j in os.listdir(args.style_dir+"/"+i):
#         paths.append(args.style_dir+"/"+i+"/"+j) 
#         #print(paths)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([
                              {'params': network.module.decoder.parameters()},
                              {'params': network.module.sa_module.parameters()}], lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    # save_image("c.jpg",content_images[0])
    # save_image("s.jpg",sontent_images[0])
    #print(content_images.size())
    loss_n,loss_c, loss_s,l_identity1, l_identity2, loss_tv= network(content_images, style_images)
    #loss_c, loss_s,l_identity1, l_identity2, loss_tv= network(content_images, style_images)

    #loss_c, loss_s= network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_tv = 10e-5 * loss_tv
    loss =  3000 * loss_n + loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) +loss_tv
    #loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) +loss_tv

    #loss = loss_c + loss_s 
    # print(loss.sum().cpu().detach().numpy(),"-n:",loss_n.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
    #           ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy(),loss_tv.sum().cpu().detach().numpy()
    #           )
    print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
              ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy(),loss_tv.sum().cpu().detach().numpy()
              )
    # loss_c, loss_s = network(content_images, style_images)
    # loss_c = args.content_weight * loss_c
    # loss_s = args.style_weight * loss_s
    # loss = loss_c + loss_s 

    
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    # writer.add_scalar('loss_content', loss_c.item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.item(), i + 1)
    # # writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    # # writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    # writer.add_scalar('total_loss', loss.item(), i + 1)    

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.module.sa_module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/sa_module_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
writer.close()
#torch.cuda.empty_cache()

