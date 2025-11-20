"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
# change the multiprocessing sharing strategy to 'file_system' to avoid error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def tensorBoard(writer, model, epoch, epoch_iter, total_iters, opt):
    losses = model.get_current_losses() # OrderedDict([('segloss', 1.9367258548736572), ('seg_ce', 1.9367258548736572), ('tranloss', 35.47449493408203), ('G_GAN', 6.188931465148926), ('tran_L1', 0.5432747602462769), ('tran_ssim', 0.9954081773757935), ('D_GAN', 7.224973678588867), ('D_real', 9.969793319702148), ('D_fake', 4.480154037475586), ('G_GAN_Feat', 18.880725860595703), ('G_VGG', 8.866153717041016)])
    visuals = model.get_current_visuals()
    metrics = model.get_current_metrics()
    
    # add losses to tensorboard
    for k, v in losses.items():
        writer.add_scalar('Loss/' + k, v, total_iters)
        
    for k, v in visuals.items():
        if len(v.shape) == 3:
            v = v.unsqueeze(1)
            # norm
        if v.dtype not in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            v_min = v.min()
            v_max = v.max()
            if v_max - v_min > 1e-6:  # 避免除以零
                v = (v - v_min) / (v_max - v_min)
            else:
                v = v - v_min  # 如果所有值都相同，设为0
        writer.add_images('Image/' + k, v, global_step=total_iters, dataformats='NCHW')

    for k, v in metrics.items():
        writer.add_scalar('Metrics/' + k, v, total_iters)
    

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.no_flip = True # no flip; comment this line if results on flipped images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # # check if the model has gradient
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Layer {name} has gradient:\n{param.grad}")
    #     else:
    #         print(f"Layer {name} has no gradient(frozen)")

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # ----------------- model infernce settings ---------------------------------
    if hasattr(opt, 'flops') and opt.flops:
        model.print_flops_summary()
        exit(0)
    if hasattr(opt, 'latency') and opt.latency:
        inference_times = model.measure_module_times(num_iter=100)
        for module_name, time_ms in inference_times.items():
            print(f"{module_name}: {time_ms:.2f} ms")
        exit(0)
    # ----------------------------------------------------------------------------
    
    writer = SummaryWriter(log_dir=opt.tensorboard_dir + '/' + opt.name)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        if opt.gradNorm :
            model.get_current_epoch(epoch) # 将当前epoch传入model，以便计算t=0时刻损失
        
        for i, data in enumerate(dataset):  # inner loop within one epoch. dataset: instance of 'CustomDatasetDataLoader'.every iter, dataset return a dict
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing 将数据从dataset中提取到model类中
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            tensorBoard(writer, model, epoch, epoch_iter, total_iters, opt)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.gradNorm:
                    model.print_weights()  # 如果使用gradNorm，则打印权重
                elif opt.model == 'hvtc_gan_parallel_uncertain' and opt.uncertainty_weight:
                    model.print_weights()
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
