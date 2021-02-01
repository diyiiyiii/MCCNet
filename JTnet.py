import jittor.nn as nn
import jittor as jt
from jittor import Module
from JTfunction import normal
from JTfunction import calc_mean_std
from jittor.models import vgg
from jittor.misc import save_image
import numpy as np

jt.flags.use_cuda = 1
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d(2, 2, 0, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d(2, 2, 0, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d(2, 2, 0, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d(2, 2, 0, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MCCNet(Module):
    def __init__(self, in_dim):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim, int(in_dim), (1,1))  ##nn.Conv2d(in_Channel,out_Channel,kernal)
        self.g = nn.Conv2d(in_dim, int(in_dim), (1,1))
        self.h = nn.Conv2d(in_dim, int(in_dim), (1,1))
        self.softmax = nn.Softmax(dim=-2) 
        self.out_conv = nn.Conv2d(int(in_dim), in_dim, (1,1))
        self.fc = nn.Linear(in_dim, in_dim)

    def execute(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()  

        F_Fc_norm = self.f(normal(content_feat))   #标准化后内容feature传入第一层卷积层
        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(-1,1,H*W) 
        G_Fs = self.h(style_feat).view(-1,1,H*W) 

        G_Fs_sum = G_Fs_norm.view(B, C ,H*W).sum(-1) #算出每个channel上的均值
        FC_S = nn.bmm(G_Fs_norm, G_Fs_norm.permute(0,2,1)).view(B,C) / G_Fs_sum
        FC_S = self.fc(FC_S).view(B,C,1,1)
        out = F_Fc_norm*FC_S
        B,C,H,W = content_feat.size()
        x = jt.float(out)

        out = x.view(B,-1,H,W)

        out = self.out_conv(out)
        out = content_feat + out

        return out

class MCC_Module(Module):    
    def __init__(self, in_dim):
        super(MCC_Module,self).__init__()
        self.MCCN = MCCNet(in_dim)

    def execute(self, content_feats, style_feats):

        content_feat_4 = content_feats[-2] 
        style_feat_4 = style_feats[-2]
        Fcsc = self.MCCN(content_feat_4, style_feat_4)
        return Fcsc

class Net(Module):
    def __init__(self, encoder, decoder):
        super(Net,self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.mcc_module = MCC_Module(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        #fix the encoder   
        for name in ['enc_1','enc_2','enc_3','enc_4','enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        #extract relu1_1,relu2_1,rlu3_1,relu4_1,relu5_1 from input image
    def encode_with_intermeidate(self,input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i+1)) #获得第i层relu
            #print("输出的维度为",results[-1].size())
            results.append(func(results[-1]))  #将上一次的输出作为这一次的输入
        return results[1:] #将五个fea都输出

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean,target_mean) + self.mse_loss(input_std,target_std)

    def execute(self, content, style):

        std = jt.float(jt.random(shape=[1,]))
        t = jt.random((content.size()))

        jt.init.uniform_(std, 0.01, 0.02)
        #noise = jt.normal(mean=0, std=std[0], size=content.size()).cuda()
        #noise = jt.init.gauss(mean=0, std=std[0], size=content.size())
        noise = jt.random(content.size(), "float32", "normal") * std[0] + 0

        content_noise = content + noise

        with jt.no_grad():
            style_feats = self.encode_with_intermeidate(style)
            content_feats = self.encode_with_intermeidate(content)
            content_feats_N = self.encode_with_intermeidate(content_noise)

        Fcsc = self.mcc_module(content_feats,style_feats)
        Ics = self.decoder(Fcsc)

        #tensorboard
        self.Icstest = Ics

        Ics_feats = self.encode_with_intermeidate(Ics)
        #print("Ics_feats[-1]的维度:",Ics_feats[-1].size(),"Ics_feats[-2]的维度:",Ics_feats[-2].size())
        #print("content_feats[-1]的维度:",content_feats[-1].size(),"content_feats[-2]的维度:",content_feats[-2].size())
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                 self.calc_content_loss(normal(Ics_feats[-2]),normal(content_feats[-2]))

        for i in range(5):
            style_feats[i].requires_grad = False

        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        #total variation loss  
        y = Ics
        tv_loss = jt.sum(jt.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + jt.sum(jt.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

        Ics_N = self.decoder(self.mcc_module(content_feats_N, style_feats))
        loss_noise = self.calc_content_loss(Ics_N,Ics)

        #identity losses lambda 1 (?)
        Icc = self.decoder(self.mcc_module(content_feats, content_feats))
        Iss = self.decoder(self.mcc_module(style_feats, style_feats))

        loss_lambda1 = self.calc_content_loss(Icc,content) + \
                       self.calc_content_loss(Iss,style)

        #identity losses lambda 2
        Icc_feats = self.encode_with_intermeidate(Icc)
        Iss_feats = self.encode_with_intermeidate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + \
                       self.calc_content_loss(Iss_feats[0], style_feats[0]) 
        for i in range(1,5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])

        return loss_noise, loss_c, loss_s, loss_lambda1, loss_lambda2, tv_loss
