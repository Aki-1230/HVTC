from tkinter import Scale
from turtle import down
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.networks import get_norm_layer, get_scheduler, init_weights, init_net, ResnetBlock
from .seg_models.fcn import FCN32s, FCN16s, FCN8s, FCNs, VGGNet
from .seg_models.SegNet import SegNet, Encoder, Decoder
import numpy as np
import os
import time
from datetime import datetime

def define_RG(input_nc, output_nc, num_classes, ngf, n_downsampling, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)
    Encoder_block = Encoder(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    DecoderSeg_block = DecoderSeg(input_nc, num_classes, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    DecoderTran_block = DencoderTran(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    use_bias = norm_layer == nn.InstanceNorm2d
    ResnetTran_block = ResnetTran(n_blocks=9, dim=256, padding_type='reflect', norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)

    return init_net(Encoder_block, init_type, init_gain, gpu_ids), init_net(DecoderSeg_block, init_type, init_gain, gpu_ids), init_net(DecoderTran_block, init_type, init_gain, gpu_ids), init_net(ResnetTran_block, init_type, init_gain, gpu_ids)

def define_FCN(num_classes, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=num_classes)

    return init_net(fcn_model, init_type, init_gain, gpu_ids)

def define_SegNet(num_classes, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    SegNet_model = SegNet(num_classes=num_classes)

    return init_net(SegNet_model, init_type, init_gain, gpu_ids)

def define_Unet(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], mode='tran', num_classes=None):
    
    norm_layer = get_norm_layer(norm_type=norm)
    Unet_G = UnetG(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, mode=mode, num_classes=num_classes)

    return init_net(Unet_G, init_type, init_gain, gpu_ids)


def define_UnetDecoder(numclass, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    UnetDecoder = UnetDecoderSeg(numclass, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(UnetDecoder, init_type, init_gain, gpu_ids)

def define_ParallelNet(input_nc, output_nc, num_classes, ngf, n_downsampling, parallel_ver, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    Bottom_block = MtlBottom(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    Seg_head = SegHead(num_classes, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)

    if parallel_ver == '0_5':
        Tran_head = TranHead(output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    elif parallel_ver == '1_0':
        Tran_head = DencoderTran(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(Bottom_block, init_type, init_gain, gpu_ids), init_net(Seg_head, init_type, init_gain, gpu_ids), init_net(Tran_head, init_type, init_gain, gpu_ids)

def define_MTAN(input_nc, output_nc, n_tasks, num_classes, ngf, norm='batch', use_resblock=False, use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    net_MTAN = MTAN(input_nc, output_nc, n_tasks, num_classes, ngf, use_resblock=use_resblock, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(net_MTAN, init_type, init_gain, gpu_ids)

class MTAN(nn.Module):
    def __init__(self, input_nc, output_nc, n_tasks, num_classes, ngf, use_resblock=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Parameters:
            encoder_block (nn.ModuleList) -- construct the backbone, store the conv layers which increase dimension in VGG
            conv_block_enc (nn.ModuleList) -- construct the backbone, store the conv layers which not increase dimension and size in VGG
            encoder_att (nn.ModuleList of nn.ModuleList, 3*5) -- nested ModuleList, 3*5, 3 for 3 tasks, 5 for 5 att blocks.
            encoder_block_att (nn.ModuleList) -- define the kenerl 3 conv layers in each att block, increase dimension
    
        """
        super(MTAN, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        filter = [64, 128, 256, 512, 512]
        self.num_classes = num_classes
        self.n_tasks = n_tasks
        self.output_nc = output_nc
        self.use_resblock = use_resblock
        self.norm_layer = norm_layer

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])]) # nc: 3 -> 64, conv + bn + relu
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])]) #nc: 64 -> 64, conv + bn + relu

        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]])) # VGG中每个conv block中的第一层conv(通道数翻倍的)
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))
        
        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])]) 
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            # 定义每个conv block中尺度通道数均不变的conv layer
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))
        
        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]) # 嵌套modulelist
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(self.n_tasks):
            # 定义attention模块中的两个 1x1 conv layer
            if j < self.n_tasks - 1:
                # 定义第 2, 3 任务的 第一个 attention block
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                # 定义第 2-5 个attention block
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))
        
        for i in range(4): # define the kenerl 3 conv layers in each att block
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
        
        self.pred_tran_head = self.conv_layer([filter[0], self.output_nc], pred=True)
        self.pred_seg_head = self.conv_layer([filter[0], self.num_classes], pred=True)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        if use_resblock: # define the resnet blocks
            n_blocks = 9 
            padding_type='reflect'
            
            self.resnet = []
            for i in range(n_blocks):       
                self.resnet += [ResnetBlock(ngf * (2 ** 4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
            self.downconv = [nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), 
                        self.norm_layer(1024), nn.ReLU(inplace=True)]
            self.upconv = [nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                      self.norm_layer(512), nn.ReLU(inplace=True)]

            self.resnet9block = nn.Sequential(*self.resnet)
            self.downconv = nn.Sequential(*self.downconv)
            self.upconv = nn.Sequential(*self.upconv)
    
    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block
        
    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):

        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * self.n_tasks for _ in range(2))
        for i in range(self.n_tasks):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(self.n_tasks):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))   # shape: [n_tasks, 5, 3]
        
        # define global shared network
        for i in range(5):
            # forward of backbone encoder part
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x) # nc 3 -> 64
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0]) # 
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        
        if self.use_resblock: # forward in resnet9block
            g_maxpool[-1] = self.downconv(g_maxpool[-1])
            g_maxpool[-1] = self.resnet9block(g_maxpool[-1])
            g_maxpool[-1] = self.upconv(g_maxpool[-1])

        for i in range(5):
            # forward of backbone decoder part
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(self.n_tasks):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])    # calculate attention mask {a}
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1] # element-wise multiplication -> {a_^}
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)) # attention mask {a}
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]  # element-wise multiplication -> {a_^}
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        tran_pred = torch.tanh(self.pred_tran_head(atten_decoder[0][-1][-1]))
        seg_pred = self.pred_seg_head(atten_decoder[1][-1][-1])

        return [tran_pred, seg_pred]

class Encoder(nn.Module):
   

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_blocks = 9
        padding_type='reflect'
        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.encoder = nn.Sequential(*encoder)

        self.visual = []

    def get_target_layer(self, index=4):
        # 这个方法用于获取特定layer，主要服务于gradnorm的计算
        # index是索引号，默认为8，表示第8个残差块
        target_layer = self.encoder[index]
        print(f'target_layer:{target_layer}')

        return target_layer
    
    def _save_mid_feature(self, module, input, output):
        self.mid_feature = output.detach().cpu().clone()
        mid_feature = self.mid_feature.numpy()
        self.visual.append(mid_feature)

    def register_hook(self):
        target_layer = self.get_target_layer()
        self.hook = target_layer.register_forward_hook(self._save_mid_feature)

    def remove_hook(self):
        self.hook.remove()

    def forward(self, input):
        """Standard forward"""
        return self.encoder(input)

class MtlBottom(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        the Parallel model's shared bottom.
        consist of 2 modules: downsampler and resnets
        input: 3*256*256
        output: 1024*16*16  (in case of n_dsp=4)

        '''
        super(MtlBottom, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_blocks = 9 # set resnet blocks
        padding_type='reflect'
        downsampler = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            downsampler += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        self.resnet = []
        for i in range(n_blocks):       # add ResNet blocks
            self.resnet += [ResnetBlock(ngf * (2 ** n_downsampling), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        self.model = nn.Sequential(*downsampler, *self.resnet)
    
    def get_target_layer(self, index=8):
        # 这个方法用于获取特定layer，主要服务于gradnorm的计算
        # index是索引号，默认为8，表示第8个残差块
        target_block = self.resnet[index]
        target_layer = target_block.conv_block[6]

        # print('target_block:',target_block)
        # print('target_layer:',target_layer)
        # input('pause')

        return target_layer

    def forward(self, input):
        # standard forward
        return self.model(input)

class SegHead(nn.Module):

    def __init__(self, num_classes, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SegHead, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        seg_head =[]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            seg_head += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        seg_head += [nn.ReflectionPad2d(3)]
        seg_head += [nn.Conv2d(ngf, num_classes, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*seg_head)
    
    def forward(self, input):
        return self.model(input)

class TranHead(nn.Module):

    def __init__(self, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(TranHead, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        tran_head =[]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            tran_head += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        tran_head += [nn.ReflectionPad2d(3)]
        tran_head += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        tran_head += [nn.Tanh()]

        self.model = nn.Sequential(*tran_head)
    
    def forward(self, input):
        return self.model(input)



class DecoderSeg(nn.Module):

    def __init__(self, input_nc, num_classes, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(DecoderSeg, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        convs = []
        mult = 2
        ### 64 * 4----> 64 * 16
        convs += [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf * (2 ** n_downsampling), ngf * mult * 16, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf * mult * 16),
                 nn.ReLU(True)]
        convs += [nn.Conv2d(ngf * mult * 16, ngf * mult * 16, kernel_size=1, padding=0, bias=use_bias),
                 norm_layer(ngf * mult * 16),
                 nn.ReLU(True)]               
        convs += [nn.Conv2d(ngf * mult * 16, num_classes, kernel_size=1)]
        
        decoder_seg = []
        for i in range(n_downsampling):
            decoder_seg += [nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias)]

        self.convs = nn.Sequential(*convs)
        self.decoder_seg = nn.Sequential(*decoder_seg)


    def forward(self, input):
        """Standard forward"""

        return self.decoder_seg(self.convs(input))        


class DencoderTran(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(DencoderTran, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_blocks = 9
        padding_type='reflect'
 
        resnet = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        decoder_tran = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            decoder_tran += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder_tran += [nn.ReflectionPad2d(3)]
        decoder_tran += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder_tran += [nn.Tanh()]

        self.resnet = nn.Sequential(*resnet)
        self.decoder_tran = nn.Sequential(*decoder_tran)

        # init a dict for visualiation
        self.visual = []

    def get_target_layer(self, index=9):
        # 这个方法用于获取特定layer，主要服务于gradnorm的计算
        # index是索引号，默认为8，表示第8个残差块
        # target_block = self.resnet[index]
        # target_layer = target_block.conv_block[6]

        target_layer = self.decoder_tran[index]
        print(f'target_layer {target_layer}')
        return target_layer

    def _save_mid_feature(self, module, input, output):
        self.mid_feature = output.detach().cpu().clone()
        mid_feature = self.mid_feature.numpy()
        self.visual.append(mid_feature)

    def register_hook(self):
        target_layer = self.get_target_layer()
        self.hook = target_layer.register_forward_hook(self._save_mid_feature)

    def remove_hook(self):
        self.hook.remove()

    def forward(self, input):
        """Standard forward"""

        return self.decoder_tran(self.resnet(input))

class ResnetTran(nn.Module):
    def __init__(self, n_blocks, dim, padding_type, norm_layer, use_dropout, use_bias) -> None:
        super(ResnetTran, self).__init__()
        resnet = []
        for i in range(n_blocks):
            resnet += [ResnetBlock(dim, padding_type, norm_layer, use_dropout, use_bias)]
        self.resnet = nn.Sequential(*resnet)

    def forward(self, input):
        return self.resnet(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out        

class GradNorm(nn.Module):
    def __init__(self, num_tasks, alpha=0.5):
        '''
        

        '''
        super(GradNorm, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = torch.nn.Parameter(torch.ones(num_tasks).float(), requires_grad=True)  # learnable weights define
    
    def weights_renorm(self):
        normalize_coeff = self.num_tasks / torch.sum(self.weights.data)
        self.weights.data = self.weights.data * normalize_coeff

    def forward(self, task_losses, target_module, current_epoch):
        # 存储初始时刻损失
        if current_epoch == 1:
            initial_task_loss = []
            initial_task_loss.append(task_losses[0]) # loss_G_GAN
            initial_task_loss.append(task_losses[1]) # loss_tran_L1
            initial_task_loss.append(task_losses[2])  # loss_tran_ssim
            initial_task_loss.append(task_losses[3])  # loss_seg_ce
            initial_task_loss.append(task_losses[4])  # loss_GAN-feat
            initial_task_loss.append(task_losses[5])  # loss_VGG
            self.initial_task_loss = torch.stack(initial_task_loss)  # size: (1,4), 存储初始时刻损失

        # 获取目标共享层
        self.W_layer = target_module.module.get_target_layer()

        # 计算梯度范数
        norms=[]
        self.task_loss = torch.stack(task_losses)

        gygw1 = torch.autograd.grad(task_losses[0], self.W_layer.parameters(), retain_graph=True)
        norms.append(torch.norm(torch.mul(self.weights[0],gygw1[0]), p=2))
        gygw2 = torch.autograd.grad(task_losses[1], self.W_layer.parameters(),retain_graph=True)
        norms.append(torch.norm(torch.mul(self.weights[1],gygw2[0]), p=2))
        gygw3 = torch.autograd.grad(task_losses[2], self.W_layer.parameters(),retain_graph=True)
        norms.append(torch.norm(torch.mul(self.weights[2],gygw3[0]), p=2))
        gygw4 = torch.autograd.grad(task_losses[3], self.W_layer.parameters(),retain_graph=True)
        norms.append(torch.norm(torch.mul(self.weights[3],gygw4[0]), p=2))
        gygw5 = torch.autograd.grad(task_losses[4], self.W_layer.parameters(),retain_graph=True)
        norms.append(torch.norm(torch.mul(self.weights[4],gygw5[0]), p=2))
        gygw6 = torch.autograd.grad(task_losses[5], self.W_layer.parameters())
        norms.append(torch.norm(torch.mul(self.weights[5],gygw6[0]), p=2))
        norms = torch.stack(norms)

        # 计算反向学习速率
        loss_ratio = []
        for i in range(self.num_tasks):
            loss_ratio.append(self.task_loss[i] / self.initial_task_loss[i])
        loss_ratio = torch.stack(loss_ratio)
        inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

        # 计算梯度标准值
        mean_norm = torch.mean(norms)

        # 计算grad_loss
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False) # 计算常数项，存为四份，torch.Size([4])
        grad_loss = torch.sum(torch.abs(norms - constant_term))

        return grad_loss
