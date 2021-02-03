import jittor as jt
import jittor.nn as nn
import jittor.transform as transforms
from jittor.misc import save_image

from JTfunction import calc_mean_std, normal, coral
import JTnet as net

import torch
import numpy as np
import cv2
import time
import argparse
import os
from PIL import Image
from os.path import basename
from os.path import splitext

jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU

#测试图预处理
def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#风格图预处理
def style_transform(h, w):
    k = (h, w)
    size = int(np.max(k))
    transform_list = []

    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#内容图预处理
def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#风格迁移
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    style_fs, content_f, style_f = feat_extractor(vgg, content, style)
    Fccc = sa_module(content_f, content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = jt.float(jt.random((1, C, H, W)))
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i+1]
        Fccc = Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)

    feat = feat * alpha + Fccc * (1 - alpha)
    #feat_norm = normal(feat)
    feat = feat

    return decoder(feat)

#加载图片
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    return paths

#加载参数文件
def load_weight(model, path):
    if '.pth' in path:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(jt.load(path))

#获取feat
def feat_extractor(vgg, content, style):
    enc_layers = list(vgg.children())
    enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

    Content3_1 = enc_3(enc_2(enc_1(content)))
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)
    Style3_1 = enc_3(enc_2(enc_1(style)))
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    content_f = [Content3_1, Content4_1, Content5_1]
    style_f = [Style3_1, Style4_1, Style5_1]
    style_fs = [enc_1(style), enc_2(enc_1(style)), enc_3(enc_2(enc_1(style))), Style4_1, Style5_1]

    return style_fs, content_f, style_f

#
def image_process(content, style):
    content_tf1 = content_transform()
    content_frame = content_tf1(content)
    content_frame = jt.array(content_frame)

    h, w, c = np.shape(content_frame)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))
    style = jt.array(style)

    if preserve_color:
        style = coral(style, content)

    style = style.unsqueeze(0)
    content = content_frame.unsqueeze(0)

    with jt.no_grad():
        output = style_transfer(vgg, decoder, mcc_module, content, style, alpha)
    output = output.squeeze(0)

    return output

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,default="./content/blonde_girl.jpg",
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default="./style/candy.jpg",
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg_path', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='./experiments/decoder_iter_160000.pth')
parser.add_argument('--transform_path', type=str, default='./experiments/mcc_module_iter_160000.pth')
parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
args = parser.parse_args()
# Additional options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
# Advanced options
preserve_color='store_true'
alpha=args.a
interpolation_weights = args.style_interpolation_weights
preserve_color = False
do_interpolation = False

#加载测试数据，内容图+风格图
if os.path.isdir(args.content_dir):
    content_paths = get_files(args.content_dir)
else:  #单一图片
    content_paths = [args.content_dir]
if os.path.isdir(args.style_dir):
    style_paths = get_files(args.style_dir)
else:  #单一风格
    style_paths = [args.style_dir]

if not os.path.exists(output_path):
    os.mkdir(output_path)

decoder = net.decoder
vgg = net.vgg
network = net.Net(vgg, decoder)
mcc_module = network.mcc_module

decoder.eval()
mcc_module.eval()
vgg.eval()

load_weight(decoder,args.decoder_path)
load_weight(mcc_module,args.transform_path)
load_weight(vgg,args.vgg_path)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)


for content_path in content_paths:
    if do_interpolation:
        style = jt.stack([style_tf(Image.open(p)) for p in style_paths])
        #print(style.size())
        content = content_tf(Image.open(content_path)) \
                  .unsqueeze(0).expand_as(style)

        with jt.no_grad():
            output = style_transfer(vgg, decoder, mcc_module, content, style, alpha, interpolation_weights)

        output_name = '{:s}/{:s}_interpolation_{:s}'.format(
            output_path, splitext(basename(content_path))[0], save_ext)
        save_image(output, output_name)

    else:  #process one content and one style
        outfile = output_path + '/' + splitext(basename(content_path))[0] + '/'
        if not os.path.exists(outfile):
            os.makedirs(outfile)

        if 'mp4' in content_path:
            for style_path in style_paths:
                video = cv2.VideoCapture(content_path)
                j = 0
                rate = video.get(5)

                width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) #获得帧宽和帧高
                fps = int(rate)

                video_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
                    splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.mp4')
                #print(video_name)
                videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (int(width), int(height)))

                while(video.isOpened()):
                    j = j+1
                    ret, frame = video.read()
                    if ret == False :
                        break

                    if j%1 == False:
                        # 对每一帧进行风格化。
                        style = Image.open(style_path)
                        content = Image.fromarray(cv2.cvColor(frame, cv2.COLOR_BGR2RGB))
                        output = image_process(frame,style)
                        # 对风格化后的结果进行额外处理，以存储到视频中
                        output = output * 255 + 0.5
                        output = jt.uint8(jt.clamp(output,0,255).permute(1, 2, 0)).numpy()
                        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                        videoWriter.write(output)  # 写入帧图
        else:
            for style_path in style_paths:
                image_name = outfile +'/{:s}_stylized_{:s}{:s}'.format(
                        splitext(basename(content_path))[0],splitext(basename(style_path))[0], '.jpg')

                content = Image.open(content_path)
                style = Image.open(style_path)
                output = image_process(content, style)
                save_image(output, image_name)

