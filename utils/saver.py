import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.cp_dir = os.path.join(args.checkpoint_dir + "-" + args.dataset + "-" + args.checkname)
        if not os.path.exists(self.cp_dir):
            os.makedirs(self.cp_dir)

    def save_checkpoint(self, state, is_best, filename=None, phase=None):
        """Saves checkpoint to disk"""
        if phase == "train":
            filename = os.path.join(self.cp_dir, filename)
            torch.save(state, filename)
        
        elif phase == "valid":
            if is_best:
                best_pred = state['best_pred']
                with open(os.path.join(self.cp_dir, 'best_pred.txt'), 'w') as f:
                    f.write(str(best_pred))
                shutil.copyfile(os.path.join(self.cp_dir, filename), os.path.join(self.cp_dir, 'model_best.pth.tar'))


    def save_experiment_config(self):
        logfile = os.path.join(self.cp_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()