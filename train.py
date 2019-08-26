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
from collections import OrderedDict 
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False
import torch
import torchvision
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torchsummary import summary

if __name__ == '__main__':
# execute only if run as a script
    opt = TrainOptions().parse()   # get training options
    # to save training options to tensorboard
    writer = SummaryWriter('runs/' + opt.TBoardX)
    for arg in vars(opt):
        #start = "| "
        #content =   start +  arg +  " | "+ str(getattr(opt, arg))+ " |"
        #print( content)
        # tensorboardx 
        writer.add_text(arg, str(getattr(opt, arg)))
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    opt.phase='val'  
    print(opt.phase)
    dataset_val = create_dataset(opt)
    dataset_val_size = len(dataset_val)    # get the number of images in the VAL dataset.
    
    print('The number of validating images = %d' % dataset_val_size)
    
    opt.phase='train'   

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)             # regular setup: load and print networks; create schedulers

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        val_acc_history = []

        for name in model.loss_names:
            globals()[name + '_sum'] = 0 
        running_corrects_train = torch.tensor(0)
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses() 

            output = model.output 
            target = model.target
            _, preds = torch.max(output, 1)
            running_corrects_train += torch.sum(preds == target)
            
            # sum of each loss for per batch, 
            for name in model.loss_names:
                globals()[name+"_sum"] = globals()[name+"_sum"] + losses[name]             
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        epoch_acc_train= running_corrects_train.double() / dataset_size
       
        print('{} Acc: {:.4f}'.format(opt.phase, epoch_acc_train))
        writer.add_scalar('train_accuracy', epoch_acc_train, epoch)

        # for evaluation
        running_corrects = torch.tensor(0)


        for i, data in enumerate(dataset_val):
            model.set_input(data)
            model.test()
            output = model.output 
            target = model.target
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == target)

        epoch_acc = running_corrects.double() / dataset_val_size
        print('{} Acc: {:.4f}'.format('val', epoch_acc))

        ###############################################
        # Tensorboardx visualization
        # loss
        writer.add_scalar('validation_accuracy', epoch_acc, epoch)
        writer.add_scalar('Learning rate', model.optimizers[0].param_groups[0]['lr'], epoch)
        for name in model.loss_names:
            globals()[name+"_Ave"] = globals()[name+"_sum"] / dataset_size
            if 'A' in name: 
                writer.add_scalar('A/'+ name , globals()[name+"_Ave"] , epoch ) 
            else: 
                writer.add_scalar('B/'+ name , globals()[name+"_Ave"] , epoch )
        # metrix 
        # waiting for implementation

        ########################################################### 
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        
    writer.export_scalars_to_json("./all_scalars_reg.json")
    writer.close()  