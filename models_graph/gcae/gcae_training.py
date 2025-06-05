import os
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from functools import partial

from utils.train_utils_final import calc_reg_loss
from utils.schedulers.delayed_sched import *
from utils.schedulers.cosine_annealing_with_warmup import *

class Trainer:
    def __init__(self, args, model, loss, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  
        self.loss = loss
        if not hasattr(self.args, "ckpt_dir") or self.args.ckpt_dir is None:
            self.args.ckpt_dir = "./checkpoints" 
        

        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr,)


    def adjust_lr(self, epoch, lr=None):
        if self.scheduler is not None:
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
       

        elif self.args.lr is not None and self.args.lr_decay is not None:
            lr = lr if lr is not None else self.args.lr  
            new_lr = lr * (self.args.lr_decay ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            raise ValueError('Missing parameters for LR adjustment')
        return new_lr
    
    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args

         # Ensure `ckpt_dir` is not None
        if self.args.ckpt_dir is None:
            self.args.ckpt_dir = "./checkpoints"  # Default directory

        # Create directory if it doesn't exist
        if not os.path.exists(self.args.ckpt_dir):
            os.makedirs(self.args.ckpt_dir)

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        
        
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state


    def train(self, log=True, checkpoint_filename=None, args=None,num_epochs=1):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'
        if num_epochs is None:  
            start_epoch = self.args.start_epoch
            num_epochs = self.args.ae_epoch  
        else:
            start_epoch = 0

        self.model.train()
        self.model = self.model.to(args.device)
        for epoch in range(start_epoch, num_epochs):
            loss_list = []
            ep_start_time = time.time()
            print("Started epoch {}".format(epoch))
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                data = data_arr[0].to(args.device, non_blocking=True)
                data = data[:,0:2, :, :].to(torch.float32)   
                
                reco_data=self.model(data) 
                reco_loss = self.loss(data, reco_data)
                

                reg_loss = calc_reg_loss(self.model)
                loss = reco_loss + 1e-3 * args.alpha * reg_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

            print("Epoch {0} done, loss: {1:.7f}, took: {2:.3f}sec".format(epoch, np.mean(loss_list),
                                                                           time.time()-ep_start_time))
            new_lr = self.adjust_lr(epoch)
            print('lr: {0:.3e}'.format(new_lr))

            self.save_checkpoint(epoch, args=args, filename=checkpoint_filename)

        return checkpoint_filename, np.mean(loss_list)

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state

    def test(self, cur_epoch, ret_sfmax=False, log=True, args=None):
        self._test(cur_epoch, self.test_loader, ret_sfmax=ret_sfmax, log=log, args=args)

    def _test(self, cur_epoch, test_loader, ret_sfmax=True, log=True, args=None):
        print("Testing")
        self.model.eval()
        test_loss = 0
        output_arr = []
        for itern, data_arr in enumerate(test_loader):
            
            with torch.no_grad():
                data = data_arr[0].to(args.device)
                data = data[:,0:2, :, :].to(torch.float32) 
                output = self.model(data)

            if ret_sfmax:
                output_sfmax = output
                output_arr.append(output_sfmax.detach().cpu().numpy())
                del output_sfmax

            loss = self.loss(output, data)
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        print("--> Test set loss {:.7f}".format(test_loss))
        self.model.train()
        if ret_sfmax:
            return output_arr
        



def init_optimizer(type_str, **kwargs):
    if type_str.lower() == 'adam':
        opt_f = optim.Adam
    else:
        return None

    return partial(opt_f, **kwargs)


def init_scheduler(type_str, lr, epochs, warmup=3):
    sched_f = None
    if type_str.lower() == 'exp_decay':
        sched_f = None
    elif type_str.lower() == 'cosine':
        sched_f = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=epochs)
    elif type_str.lower() == 'cosine_warmup':
        sched_f = partial(CosineAnnealingWarmUpRestarts, T_0=epochs, T_up=warmup)
    elif type_str.lower() == 'cosine_delayed':
        sched_f = partial(DelayedCosineAnnealingLR, delay_epochs=warmup,
                          cosine_annealing_epochs=epochs)
    elif (type_str.lower() == 'tri') and (epochs >= 8):
        sched_f = partial(optim.lr_scheduler.CyclicLR,
                          base_lr=lr/10, max_lr=lr*10,
                          step_size_up=epochs//8,
                          mode='triangular2',
                          cycle_momentum=False)
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return sched_f



