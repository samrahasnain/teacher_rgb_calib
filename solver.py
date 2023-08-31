import torch
from torch.nn import functional as F
from DCF_ResNet_models import DCF_ResNet
import numpy as np
import os
import cv2
import time
import torch.nn as nn
import argparse
import os.path as osp
import os
from tqdm import trange, tqdm
import torch
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.RGBnet = DCF_ResNet()
        self.Dnet = DCF_ResNet()
       
        if self.config.cuda:
            self.RGBnet = self.RGBnet.cuda()
            self.Dnet = self.Dnet.cuda()
        self.lr = self.config.lr
        self.wd = self.config.wd

        self.RGBoptimizer = torch.optim.Adam(self.RGBnet.parameters(), lr=self.lr, weight_decay=self.wd)
        self.Doptimizer = torch.optim.Adam(self.Dnet.parameters(), lr=self.lr, weight_decay=self.wd)
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.RGBmodel)
            self.RGBnet.load_state_dict(torch.load(self.config.RGBmodel))
            print('Loading pre-trained model from %s...' % self.config.Dmodel)
            self.Dnet.load_state_dict(torch.load(self.config.Dmodel))
        if config.mode == 'train':
            if self.config.RGBload != '':
                self.RGBnet.load_state_dict(torch.load(self.config.RGBload))  # load pretrained model
                self.Dnet.load_state_dict(torch.load(self.config.Dload))  # load pretrained model
    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                atts_rgb, dets_rgb,x3_r,x4_r,x5_r = self.RGBnet(images)
                atts_depth, dets_depth, x3_d,x4_d,x5_d = self.Dnet(depth)
                print(atts_rgb.shape, dets_rgb.shape,atts_depth.shape, dets_depth.shape)
                atts_rgb = np.squeeze(torch.sigmoid(atts_rgb)).cpu().data.numpy()
                atts_rgb = (atts_rgb - atts_rgb.min()) / (atts_rgb.max() - atts_rgb.min() + 1e-8)
                atts_rgb = 255 * atts_rgb
                filename = os.path.join(self.config.test_folder_atts_rgb, name[:-4] + '_atts_rgb.png')
                cv2.imwrite(filename, atts_rgb)
                dets_rgb = np.squeeze(torch.sigmoid(dets_rgb)).cpu().data.numpy()
                dets_rgb = (dets_rgb - dets_rgb.min()) / (dets_rgb.max() - dets_rgb.min() + 1e-8)
                dets_rgb = 255 * dets_rgb
                filename = os.path.join(self.config.test_folder_dets_rgb, name[:-4] + '_dets_rgb.png')
                cv2.imwrite(filename, dets_rgb)
                atts_depth = np.squeeze(torch.sigmoid(atts_depth)).cpu().data.numpy()
                atts_depth = (atts_depth - atts_depth.min()) / (atts_depth.max() - atts_depth.min() + 1e-8)
                atts_depth = 255 * atts_depth
                filename = os.path.join(self.config.test_folder_atts_depth, name[:-4] + '_atts_depth.png')
                cv2.imwrite(filename, atts_depth)
                dets_depth = np.squeeze(torch.sigmoid(dets_depth)).cpu().data.numpy()
                dets_depth = (dets_depth - dets_depth.min()) / (dets_depth.max() - dets_depth.min() + 1e-8)
                dets_depth = 255 * dets_depth
                filename = os.path.join(self.config.test_folder_dets_depth, name[:-4] + '_dets_depth.png')
                cv2.imwrite(filename, dets_depth)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        step=0  
        loss_vals=  []
        self.RGBnet.train()
        self.Dnet.train()
        
        for epoch in range(self.config.epoch):
            loss_rgb_item = 0
            loss_depth_item=0
            for i, data_batch in tqdm(enumerate(self.train_loader)):
                sal_image, sal_depth, sal_label= data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                        'sal_label']

                
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)
                
                step+=1
                self.RGBoptimizer.zero_grad()
                self.Doptimizer.zero_grad()
                atts_rgb, dets_rgb,x3_r,x4_r,x5_r= self.RGBnet(sal_image)
                atts_depth, dets_depth, x3_d,x4_d,x5_d= self.Dnet(sal_depth)
                loss1_rgb = F.binary_cross_entropy_with_logits(atts_rgb, sal_label, reduction='sum')
                loss2_rgb = F.binary_cross_entropy_with_logits(dets_rgb, sal_label, reduction='sum')
                loss_rgb = (loss1_rgb + loss2_rgb) / 2.0
                loss_rgb_item += loss_rgb.item() * sal_image.size(0)
                loss_rgb.backward()
                loss1_depth = F.binary_cross_entropy_with_logits(atts_depth, sal_label, reduction='sum')
                loss2_depth = F.binary_cross_entropy_with_logits(dets_depth, sal_label, reduction='sum')
                loss_depth = (loss1_depth + loss2_depth) / 2.0
                loss_depth_item += loss_depth.item() * sal_depth.size(0)
                loss_depth.backward()
          
                self.RGBoptimizer.step()
                self.Doptimizer.step()


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.RGBnet.state_dict(), '%s/DCF_rgb_epoch_%d.pth' % (self.config.save_folder_rgb, epoch + 1))
                torch.save(self.Dnet.state_dict(), '%s/DCF_D_epoch_%d.pth' % (self.config.save_folder_depth, epoch + 1))
            train_loss_rgb = loss_rgb_item/len(self.train_loader.dataset)
            train_loss_depth = loss_depth_item/len(self.train_loader.dataset)
          
               
            print('Epoch:[%2d/%2d] | Train Loss RGB: %.3f | Train Loss Depth : %0.3f' % (epoch, self.config.epoch,train_loss_rgb,train_loss_depth))
                
                
        
            # save model
        torch.save(self.RGBnet.state_dict(), '%s/final.pth' % self.config.save_folder_rgb)
        torch.save(self.Dnet.state_dict(), '%s/final.pth' % self.config.save_folder_depth)
            
    

            
        
