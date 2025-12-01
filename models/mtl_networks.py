import torch
import torch.nn as nn
from models.networks import get_norm_layer, init_net, ResnetBlock
from models.hvtc_networks import TranHead, SegHead, MtlBottom
import functools

class ConvEncoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        the Parallel model's shared bottom.
        consist of 2 modules: downsampler and resnets
        input: 3*256*256
        output: 1024*16*16  (in case of n_dsp=4)

        '''
        super(ConvEncoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_blocks = 9 # set resnet blocks
        padding_type='reflect'

        self.model = nn.ModuleList([])
        self.model.append(
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)
            )
        )

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            self.model.append(
                nn.Sequential(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)
                )
            )
        
        self.resnet = []
        for i in range(n_blocks):       # add ResNet blocks
            self.resnet += [ResnetBlock(ngf * (2 ** n_downsampling), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model.append(nn.Sequential(*self.resnet))
        
    

    def forward(self, input):
        # standard forward
        print('network error, cannot forward directly')
        return

    def forward_stage(self, input, stage):
        # stage -> 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'
        idx = int(stage[-1]) - 1
        return self.model[idx](input)

def define_CrossStitchNet(input_nc, output_nc, num_classes, ngf, n_downsampling, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    tran_encoder = ConvEncoder(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    seg_encoder = ConvEncoder(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    Seg_head = SegHead(num_classes, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    Tran_head = TranHead(output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    return (
            init_net(tran_encoder, init_type, init_gain, gpu_ids),
            init_net(seg_encoder, init_type, init_gain, gpu_ids),
            init_net(Seg_head, init_type, init_gain, gpu_ids), 
            init_net(Tran_head, init_type, init_gain, gpu_ids)
            )

def define_ParallelNet(input_nc, output_nc, num_classes, ngf, n_downsampling, parallel_ver, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    Bottom_block = MtlBottom(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    Seg_head = SegHead(num_classes, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)

    if parallel_ver == '0_5':
        Tran_head = TranHead(output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    elif parallel_ver == '1_0':
        Tran_head = DencoderTran(input_nc, output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)

    return init_net(Bottom_block, init_type, init_gain, gpu_ids), init_net(Seg_head, init_type, init_gain, gpu_ids), init_net(Tran_head, init_type, init_gain, gpu_ids)

class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels, alpha=0.9, beta=0.1):
        super().__init__()
        self.num_tasks, self.tasks = len(tasks), tasks

        self.A = nn.Parameter(torch.zeros(self.num_tasks, self.num_tasks, num_channels))

        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i == j:
                    self.A.data[i, j].fill_(alpha)
                else:
                    self.A.data[i, j].fill_(beta)
    
    def forward(self, task_features): # task_features: dict{'tran': tensor, 'seg': tensor}
        outs = {}
        # A_norm = torch.softmax(self.A, dim=1)

        for i, t_i in enumerate(self.tasks):
            out_i = 0
            for j, t_j in enumerate(self.tasks):
                w = self.A[i, j].view(1, -1, 1, 1)
                out_i = out_i + w * task_features[t_j]
            outs[t_i] = out_i

        return outs

class sShare_Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(sShare_Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        shared_conv_layers, layers = 2, []

        layers += [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)] # conv 0

        for i in range(shared_conv_layers): # conv 1 ~ conv 2
            mult = 2 ** i
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

class task_Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_downsampling, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(task_Encoder, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        share_conv_blocks, n_blocks = 3, 9
        self.layers = nn.ModuleList([])

        # conv blocks
        for i in range(2, 4): # conv 3 ~ conv 4
            mult = 2 ** i
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)
                )
            )

        # resnet blocks
        padding_type='reflect'
        resnet = []
        for i in range(n_blocks):
            resnet += [ResnetBlock(ngf * (2 ** n_downsampling), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.layers.append(nn.Sequential(*resnet))

    def forward(self, input):
        # standard forward
        print('network error, cannot forward directly')
        return

    def forward_stage(self, input, stage):
        '''
        self.layers[0]: conv3,
        self.layers[1]: conv4,
        self.layers[2]: resnet,
        stage: 'conv3'', 'conv4', 'resnet'
        '''
        map = {'conv3': 0, 'conv4': 1, 'resblocks': 2}
        return self.layers[map[stage]](input)

class tran_Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, n_downsampling=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(tran_Decoder, self).__init__()

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

class seg_Decoder(nn.Module):
    def __init__(self, num_classes, n_downsampling, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(seg_Decoder, self).__init__()

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

def define_sShareNet(input_nc, output_nc, num_classes, ngf, n_downsampling, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    share_encoder = sShare_Encoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    tran_encoder = task_Encoder(input_nc, output_nc, n_downsampling, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    seg_encoder = task_Encoder(input_nc, output_nc, n_downsampling, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    TranHead = tran_Decoder(output_nc, ngf, n_downsampling, norm_layer=norm_layer, use_dropout=use_dropout)
    SegHead = seg_Decoder(num_classes, n_downsampling, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return (
        init_net(share_encoder, init_type, init_gain, gpu_ids),
        init_net(tran_encoder, init_type, init_gain, gpu_ids),
        init_net(seg_encoder, init_type, init_gain, gpu_ids),
        init_net(TranHead, init_type, init_gain, gpu_ids),
        init_net(SegHead, init_type, init_gain, gpu_ids)
    )