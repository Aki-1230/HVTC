import torch
from .base_model import BaseModel
from . import hvtc_networks
from . import networks
from . import mtl_networks
from losses.segloss import DiceLoss, CrossEntropyLoss2d
from losses.ssimloss import SSIM
import itertools


class hvtcGANCrossStitchModel(BaseModel):
    '''
    添加组件：
    1. GAN feat loss
    2. VGG loss
    3. multicale Discriminator
    4. cross-stitch units
    5. seg module: FCN

    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='segaligned', n_epochs=100, n_epochs_decay=100, lr=0.0002, n_downsampling=4)
        parser.set_defaults(alpha=0.9, beta=0.1) # init set for cross-stitch
        parser.add_argument('--num_classes', type=int, help='number of classes for segmentation')
        parser.add_argument('--parallel_ver', type=str, default='0_5', help='parallel version')
        if is_train:
            parser.set_defaults(batch_size=2)
            parser.set_defaults(netD='multiscale')
            parser.add_argument('--gan_feat_loss', action='store_true', default=True, help='whether to use feature loss')
            parser.add_argument('--vgg_loss', action='store_true', default=True, help='whether to use vgg perception loss')
            parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for vgg feature matching loss')
            parser.add_argument('--stitch_lr_scale', type=float, default=5.0, help='learning rate scale for cross-stitch')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.tasks = ['tran', 'seg']
        self.stages = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['segloss', 'seg_ce']
        self.loss_names += ['tranloss', 'G_GAN', 'tran_L1', 'tran_ssim', 'D_GAN',
                            'D_real', 'D_fake']
        self.metric_names = ['SSIM', 'PSNR']
        
        if self.isTrain:
            if opt.gan_feat_loss:
                self.loss_names += ['G_GAN_Feat']
            if opt.vgg_loss:
                self.loss_names += ['G_VGG']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'pred_label','real_label']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['EC_backbone', 'Seg_head', 'Tran_head', 'D', 'Cross_Stitch']
        else:  # during test time, only load G
            self.model_names = ['EC_backbone', 'Seg_head', 'Tran_head', 'Cross_Stitch']

        
        if opt.netD == 'multiscale':
            self.isMultiScale_D = True
        
        if self.isTrain:
            self.gan_feat_loss = opt.gan_feat_loss
            self.vgg_loss = opt.vgg_loss
            self.n_layers_D = opt.n_layers_D
            self.num_D = opt.num_D
            self.lambda_feat = opt.lambda_feat

        
        self.num_classes = self.opt.num_classes

        # self.netEC_backbone -> nn.ModuleDict, self.netSeg_head, self.netTran_head -> nn.Module
        self.netEC_Tran, self.netEC_Seg, self.netSeg_head, self.netTran_head = mtl_networks.define_CrossStitchNet(opt.input_nc, opt.output_nc, 
                                                                                                  opt.num_classes, opt.ngf, 
                                                                                                  opt.n_downsampling, 
                                                                                                  opt.parallel_ver, opt.norm, 
                                                                                                  not opt.no_dropout, opt.init_type, 
                                                                                                  opt.init_gain, self.gpu_ids)


        self.netEC_backbone = torch.nn.ModuleDict({
            'tran': self.netEC_Tran,
            'seg': self.netEC_Seg
        })

        self.channel = {'conv1': 64, 'conv2': 128, 'conv3': 256, 'conv4': 512, 'conv5': 1024}
        self.netCross_Stitch = torch.nn.ModuleDict({stage: mtl_networks.CrossStitchUnit(self.tasks, self.channel[stage], opt.alpha, opt.beta) for stage in self.stages})
        for module in self.netCross_Stitch.values():
            module.to(self.gpu_ids[0])

        self.netEC = torch.nn.ModuleDict({
            'backbone': self.netEC_backbone,
            'cross_stitch': self.netCross_Stitch
        })

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            # define loss functions
            self.crossloss = CrossEntropyLoss2d()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.ssimloss = SSIM(window_size=11)
            if self.gan_feat_loss:
                self.criterionFeat = torch.nn.L1Loss()
            if self.vgg_loss:    
                self.criterionVGG = networks.VGGLoss(opt.gpu_ids[0])
            

            self.optimizer_G = torch.optim.Adam([
                {'params': itertools.chain(self.netEC_backbone.parameters(), self.netSeg_head.parameters(), self.netTran_head.parameters()), 'lr': opt.lr},
                {'params': self.netCross_Stitch.parameters(), 'lr': opt.lr * opt.stitch_lr_scale}
            ], lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.SGD([
            #     {'params': itertools.chain(self.netEC_backbone.parameters(), self.netSeg_head.parameters(), self.netTran_head.parameters()), 'lr': opt.lr},
            #     {'params': self.netCross_Stitch.parameters(), 'lr': opt.lr * opt.stitch_lr_scale}
            # ], lr=opt.lr, momentum=0.9)
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_label = input['label'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # print(self.real_A.shape)

    def onehot_label(self, segout):

        out = segout.argmax(dim=1).unsqueeze(1)
        out = out * 0.1  ## label: 0, 1, 2, ..., 7 ---> 0, 0.1, 0.2, ..., 0.7
        out = (out - 0.5) / 0.5
        return out

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        x = {task: self.real_A for task in self.tasks}
        for stage in self.stages:
            for task in self.tasks:
                x[task] = self.netEC_backbone[task].module.forward_stage(x[task], stage)
            
            # cross stitch
            x = self.netCross_Stitch[stage](x)

        self.fake_B = self.netTran_head(x['tran'])
        self.pred_map = self.netSeg_head(x['seg'])
        self.pred_label = self.onehot_label(self.pred_map)

        self.compute_metrics()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_GAN = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D_GAN.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_tran_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_tran_ssim = (1 - self.ssimloss(self.fake_B, self.real_B))
        self.loss_tranloss = self.loss_G_GAN + self.loss_tran_L1 + self.loss_tran_ssim

        self.loss_seg_ce = self.crossloss(self.pred_map, self.real_label)
        self.loss_segloss = self.loss_seg_ce

        # GAN feature matching loss
        if self.gan_feat_loss:
            self.loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            pred_real = self.netD(self.real_B)
            for i in range(self.num_D):
                for j in range(len(pred_fake[i])-1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
            self.loss_tranloss += self.loss_G_GAN_Feat

        # vgg loss
        if self.vgg_loss:
            self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat
            self.loss_tranloss += self.loss_G_VGG

        self.loss_G = self.loss_tranloss + self.loss_segloss
        self.loss_G.backward()
        

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def compute_metrics(self):
        import torch.nn.functional as F
        from torchmetrics import StructuralSimilarityIndexMeasure

        def compute_psnr(img1, img2, max_val=1.0):
            mse = F.mse_loss(img1, img2)
            if mse == 0:
                return torch.tensor(100.0)
            psnr_val = 10 * torch.log10(max_val ** 2 / mse)
            return psnr_val
        
        def compute_ssim(img1, img2, max_val=1.0):
            ssim = StructuralSimilarityIndexMeasure(data_range=max_val).to(self.device)
            return ssim(img1, img2)
        
        self.metric_PSNR = compute_psnr(self.fake_B, self.real_B)
        self.metric_SSIM = compute_ssim(self.fake_B, self.real_B)
    
    def print_cross_stitch_debug_info(self):
        """打印Cross-Stitch的完整调试信息"""
        print("\n" + "="*50)
        print("Cross-Stitch Debug Info")
        print("="*50)
        
        for stage, module in self.netCross_Stitch.items():
            print(f"\nStage: {stage}")
            for name, param in module.named_parameters():
                # 参数值信息
                param_data = param.data
                print(f"  {name}:")
                print(f"    Value - mean: {param_data.mean():.6f}, std: {param_data.std():.6f}")
                print(f"    Value - min: {param_data.min():.6f}, max: {param_data.max():.6f}")
                
                # 检查异常值
                if torch.isnan(param_data).any():
                    print(f"    ⚠️  WARNING: Contains NaN values!")
                if torch.isinf(param_data).any():
                    print(f"    ⚠️  WARNING: Contains Inf values!")
                
                # 梯度信息
                if param.grad is not None:
                    grad_data = param.grad
                    print(f"    Gradient - norm: {grad_data.norm():.6f}, mean: {grad_data.mean():.6f}")
                    
                    if torch.isnan(grad_data).any():
                        print(f"    ⚠️  WARNING: Gradient contains NaN!")
                    if torch.isinf(grad_data).any():
                        print(f"    ⚠️  WARNING: Gradient contains Inf!")
                else:
                    print(f"    Gradient - None (可能未反向传播)")
        
        print("="*50)