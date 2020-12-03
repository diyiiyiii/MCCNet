import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net as net
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import cv2
import time
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    #print(type(size))
    transform_list = []
     
    transform_list.append(transforms.Resize(size))
    
    #transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def content_transform():
    
    transform_list = []
     
    
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    style_fs, content_f, style_f=feat_extractor(vgg, content, style)
   # print(content_f[4])
    Fccc = sa_module(content_f,content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc=Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    #print(type(feat),type(Fccc))
    feat = feat * alpha + Fccc * (1 - alpha)
    feat_norm = normal(feat)
    feat = feat 
    return decoder(feat)
  
def feat_extractor(vgg, content, style):
  norm = nn.Sequential(*list(vgg.children())[:1])
  enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
  enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
  enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
  enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
  enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

  norm.to(device)
  enc_1.to(device)
  enc_2.to(device)
  enc_4.to(device)
  enc_5.to(device)
  content3_1 = enc_3(enc_2(enc_1(content)))
  Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
  Content5_1 = enc_5(Content4_1)
  Style3_1 = enc_3(enc_2(enc_1(style)))
  Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
  Style5_1 = enc_5(Style4_1)
  

  content_f=[content3_1,Content4_1,Content5_1]
  style_f=[Style3_1,Style4_1,Style5_1]

 
  style_fs = [enc_1(style),enc_2(enc_1(style)),enc_3(enc_2(enc_1(style))),Style4_1, Style5_1]
  
  return style_fs,content_f, style_f
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_100000.pth')
parser.add_argument('--transform_path', type=str, default='/experiments/MCC_module_iter_100000.pth')
parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
args = parser.parse_args()
args = parser.parse_args()
content="./content/blonde_girl.jpg"
style="./style/candy.jpg"
vgg_path='./experiments/vgg_normalised.pth'


# Additional options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output

# Advanced options
preserve_color='store_true'
alpha=args.a


preserve_color=False

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Either --content or --contentDir should be given.
# assert (content or content_dir)
# # Either --style or --styleDir should be given.
# assert (style or style_dir)

# if args.content:
#     content_paths = [args.content]
# else:
#     content_dir = args.content_dir
#     content_paths = [f for f in content_dir.glob('*')]
# print(content_paths)

# if args.style:
  
#     style_paths = [Path(args.style)]

# else:
#     style_dir = Path(args.style_dir)
#     style_paths = [f for f in style_dir.glob('*')]
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    # return [os.path.join(img_dir,x) for x in files]
    return paths

if os.path.isdir(args.content_dir):
    content_paths = get_files(args.content_dir)
else: # Single image file
    content_paths = [args.content_dir]
if os.path.isdir(args.style_dir):
    style_paths = get_files(args.style_dir)
else: # Single image file
    style_paths = [args.style_dir]

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(output_path+"/content"):
    os.mkdir(output_path+"/content")
decoder = net.decoder
#decoder = net.Decoder(net.decoder)
vgg = net.vgg
network=net.Net(vgg,decoder)
mcc_module = network.mcc_module


decoder.eval()
mcc_module.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path)
for k, v in state_dict.items():
    print(k)
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.transform_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
mcc_module.load_state_dict(new_state_dict)

vgg.load_state_dict(torch.load(vgg_path))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
mcc_module.to(device)
decoder.to(device)


content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf1(Image.open(p)) for p in style_paths])
        print(style.size())
        content = content_tf1(Image.open(content_path)) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, mcc_module, content, style,
                                    alpha, interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation_{:s}'.format(
            output_path, splitext(basename(content_path))[0], save_ext)
        save_image(output, output_name)

    else:  # process one content and one style
        for style_path in style_paths:
            start = time.time()
            try:
                #print(content_path,style_path)
                # content = content_tf(Image.open(content_path).convert("RGB"))
                # style = style_tf(Image.open(style_path).convert("RGB"))
                video = cv2.VideoCapture(content_path)    
                j = 0
                rate = video.get(5)  #cap.get（）括号中的参数为5代表获取帧速率
                
                width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) #获得视频帧宽和帧高
                print(width,height)
                fps = int(rate)
                #print("--------",rate)
                video_name = output_path +'/{:s}_stylized_{:s}{:s}'.format(
                        splitext(basename(content_path))[0],splitext(basename(style_path))[0], '.mp4')
                print(video_name)
                videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)))
                # try: 
                while (video.isOpened()):
               

                    j=j+1
                    
                    #print(j)
                    ret, frame = video.read()
            
                    #print("------",frame)
                   # print(np.shape(frame))
                    #cv2.imwrite("pic/"+self.video_list[index]+str(j)+".jpg", frame)
                   
                    if j %1 ==0:
                        print(j)
                        output_name1 = '{:s}/{:s}/stylized_{:s}{:s}'.format(output_path, splitext(basename(content_path))[0],str("%04d" % j),save_ext)
                        # cv2.imwrite(output_name1, frame)
                        content_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not os.path.exists('{:s}/{:s}/'.format(output_path, splitext(basename(content_path))[0])):
                            os.makedirs('{:s}/{:s}/'.format(output_path, splitext(basename(content_path))[0]))
                        #rgb_fraame = cv2.resize(rgb_frame, self.img_shape)
                        #print(rgb_frame)
                        #print("--",np.shape(rgb_frame))
                        content_tf1 = content_transform()
                        content_frame = content_tf(Image.fromarray(content_frame))
                        #print(rgb_frame.size())
                        
                        save_image(content_frame, output_name1)
                        # h,w,c=np.shape(content_frame)
                        # style_tf1 = style_transform(h,w)
                        # style = style_tf(Image.open(style_path).convert("RGB"))
                        # if preserve_color:
                        #     style = coral(style, content)
                        # style = style.to(device).unsqueeze(0)
                        # content = content_frame.to(device).unsqueeze(0)
                        # #print("------------",content.size(),style.size())
                        # with torch.no_grad():
                        #     output = style_transfer(vgg, decoder, sa_module, content, style,
                        #                             alpha)
                        # output = output.cpu()
                      
                        # output_name = '{:s}/{:s}_stylized_{:s}_{:s}{:s}'.format(
                        #     output_path, splitext(basename(content_path))[0],
                        #     splitext(basename(style_path))[0], str(j),save_ext
                        # )
                        # #save_image(output, output_name)
                        # print(output.size())
                        # output = output.squeeze()
                        # output = output.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
       
                        # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                        

                        
                        # videoWriter.write(output)  # 写入帧图
            except:
                print("over")
                end = time.time()
                print(end-start)
                        
          
