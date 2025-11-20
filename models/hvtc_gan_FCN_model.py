import torch
from .base_model import BaseModel
from . import hvtc_networks
from . import networks
from losses.segloss import DiceLoss, CrossEntropyLoss2d
from losses.ssimloss import SSIM
import itertools


class hvtcGANFCNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='segaligned', n_epochs=100, n_epochs_decay=100, lr=0.0002, n_downsampling=2)
        parser.add_argument('--parm_segmap', type=str, default='1,1', help='weight for predict sar-segmap and opt-segmap')
        parser.add_argument('--num_classes', type=int, help='number of classes for segmentation')
        parser.add_argument('--flops', action='store_true', help='whether to compute flops' )
        parser.add_argument('--latency', action='store_true', help='whether to compute latency' )
        if is_train:
            parser.add_argument('--lambda_seg', type=str, default='30,25,30', help='weight for segmentation loss')
            parser.add_argument('--lambda_tran', type=str, default='1,10,10,0,0', help='weight for translation loss')
            parser.set_defaults(batch_size=2)
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['segloss', 'seg_ce', 'seg_ce_idt']
        self.loss_names += ['tranloss', 'G_GAN', 'tran_L1', 'tran_ssim', 'D_GAN',
                            'D_real', 'D_fake']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'pred_label', 'label_op', 'label_sar', 'idt_pred_label']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['EC_SAR', 'DC_SAR_Seg', 'DC_Tran', 'OP_Seg', 'D'] # EC: Encoder; DC: Decoder; D: Discriminator.
        else:  # during test time, only load G
            self.model_names = ['EC_SAR', 'DC_SAR_Seg', 'DC_Tran', 'OP_Seg']

        if self.isTrain:
            self.weights_seg = list(map(float, self.opt.lambda_seg.split(',')))
            # Two weights: seg_ce seg_ce_idt
            self.weights_tran = list(map(float, self.opt.lambda_tran.split(',')))
            # Five weights: G_GAN/D_GAN tran_L1 tran_ssim tran_L1_idt tran_ssim_idt   

        self.weights_segmap = list(map(float, self.opt.parm_segmap.split(',')))

        # define networks (both generator and discriminator)
        self.num_classes = self.opt.num_classes

        self.netEC_SAR, self.netDC_SAR_Seg, self.netDC_Tran, _ = hvtc_networks.define_RG(opt.input_nc, opt.output_nc,
                                                                                         self.num_classes, opt.ngf, opt.n_downsampling,
                                                                                         opt.norm,
                                                                                         not opt.no_dropout,
                                                                                         opt.init_type, opt.init_gain,
                                                                                         self.gpu_ids)
        
        self.netOP_Seg = hvtc_networks.define_FCN(self.num_classes, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            # define loss functions
            self.crossloss = CrossEntropyLoss2d()  # torch.nn.CrossEntropyLoss()
            self.diceloss = DiceLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.ssimloss = SSIM(window_size=11)
            # self.focalloss = Focal_loss(gamma=2)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netEC_SAR.parameters(), self.netDC_SAR_Seg.parameters(),
                                self.netDC_Tran.parameters(), self.netOP_Seg.parameters(),
                                ), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.SGD(self.netG.parameters(),lr=opt.lr,momentum=0.9, weight_decay=0.0005)
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

        ### input: [n,8,256,256]
        # out = torch.gt(segout,0.5)
        # out = out.long()
        out = segout.argmax(dim=1).unsqueeze(1)
        out = out * 0.1  ## label: 0, 1, 2, ..., 7 ---> 0, 0.1, 0.2, ..., 0.7
        out = (out - 0.5) / 0.5
        return out

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #### sar -> label_sar and fake_op
        feature_sar = self.netEC_SAR(self.real_A)
        self.fake_B = self.netDC_Tran(feature_sar)
        self.labelsar = self.netDC_SAR_Seg(feature_sar)
        self.label_sar = self.onehot_label(self.labelsar)  # branch 1

        ### fake_op -> label_op
        self.labelop = self.netOP_Seg(self.fake_B)
        self.label_op = self.onehot_label(self.labelop)    # branch 2

        ### label_sar + label_op -> label
        self.label = self.labelsar * self.weights_segmap[0] + self.labelop * self.weights_segmap[1]
        self.pred_label = self.onehot_label(self.label)    

        #### idt for seg
        self.labelidt = self.netOP_Seg(self.real_B)
        self.idt_pred_label = self.onehot_label(self.labelidt) # branch 3

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach()) # size: [batch_size, 1, 30, 30]
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_GAN = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D_GAN.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        ############ for tran 
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.weights_tran[0]
        self.loss_tran_L1 = self.criterionL1(self.fake_B, self.real_B) * self.weights_tran[1]
        self.loss_tran_ssim = (1 - self.ssimloss(self.fake_B, self.real_B)) * self.weights_tran[2]    # weights_tran : 1, 10, 10, 0, 0
        self.loss_tran_L1_idt = 0 * self.weights_tran[3]
        self.loss_tran_ssim_idt = 0 * self.weights_tran[4]
        self.loss_tranloss = self.loss_G_GAN + self.loss_tran_L1 + self.loss_tran_ssim + self.loss_tran_ssim_idt + self.loss_tran_L1_idt

        ############ for seg

        self.loss_seg_ce = self.crossloss(self.label, self.real_label) * self.weights_seg[0]
        self.loss_seg_ce_idt = self.crossloss(self.labelidt, self.real_label) * self.weights_seg[1]
        # self.loss_seg_ce_idt = self.crossloss(self.labelidt, self.label) * self.weights_seg[1]
        # self.loss_seg_focal = self.focalloss(self.label, self.real_label) * self.weights_seg[2]     # weights_seg : 30, 25, 30
        self.loss_segloss = self.loss_seg_ce_idt + self.loss_seg_ce


        # combine loss and calculate gradients
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
    

