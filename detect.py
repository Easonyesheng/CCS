"""load checkpoints and pred"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import cv2
import logging
from tqdm import tqdm
import glob
import re

from models.unet_model import UNet
from utils.utils import img_show, log_init, gray2heat_img, test_dir_if_not_create
from settings.settings import *

class Detect(object):
    """For high volume corner detection
    """
    def __init__(self, log_name):
        """
        """
        self.log_name = log_name
        self.logfile = 'DetectLog-'+self.log_name+'.txt'
        log_init(os.path.join(LOGFILEPATH,self.logfile))

        self.img_path = r''
        self.save_path = r''
        self.checkpoint_path = r''

        self.model_name = 'UNet'
        self.size = (480,480)
        self.norm_size = 480

    def __img_preprocess(self, pil_img, device):
        """
        Args:
            pil_img: PIL.Image
        Returns:
            img: GPU img
        """
        pil_img = pil_img.convert('L')
        pil_img = pil_img.resize(self.size)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd

        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        transform_toTensor = transforms.ToTensor()
        img = transform_toTensor(img_trans)
        img = img.unsqueeze(0)  
        img = img.to(device, dtype=torch.float32)
        return img

    def individual_detect(self, img_name, index, show_flag=False, save_flag=False ):
        """
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        img_path = img_name
        logging.info(f'load img from {img_path}')
        pil_img = Image.open(img_path)
        if show_flag:
            pil_img.show()

        img = self.__img_preprocess(pil_img, device)

        # load model
        logging.info(f'load {self.model_name}')
        if self.model_name == 'UNet':
            net = UNet(1,1)

        net.load_state_dict(torch.load(self.checkpoint_path))

        net.eval()

        net.to(device=device)

        heatmap_pred = net(img)
        heatmap_pred = heatmap_pred.squeeze()
        heatmap_np = heatmap_pred.detach().cpu().numpy()
        # print(heatmap_np.shape)

        color_img = gray2heat_img(heatmap_np)

        if show_flag:
            img_show(heatmap_np, 'heatmap')
            
        
        if save_flag:
            save_path = self.save_path
            logging.info(f'Saved in {save_path}')
            heatmap_path = os.path.join(self.save_path, 'heatmap')
            np.save(os.path.join(heatmap_path, str(index)+'.npy'), heatmap_np)
            color_path = os.path.join(self.save_path, 'color_img')
            cv2.imwrite(os.path.join(color_path, str(index)+'.jpg'), color_img)

    def high_volume_detect(self, folderpath, show_flag=False, save_flag=False):
        """
        """
        self.img_path = folderpath
        logging.info(f'load imgs from {folderpath}')
        ori_img_list = glob.glob(os.path.join(self.img_path, '*.jpg'))
        ori_img_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))
        # ori_img_list.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))

        print(ori_img_list[:10])
        l = len(ori_img_list)
        logging.info(f'load {l} imgs totally.')

        for i, img in enumerate(ori_img_list):
            index = img.split('\\')[-1].split('.')[0]
            self.individual_detect(img_name=img,index=index,show_flag=show_flag, save_flag=save_flag)
    
    def __norm_for_numpy(self, npy):
        """
        """
        max_ = npy.max()
        min_ = npy.min()
        ave = max_ - min_
        ave /= 2
        npy[np.where(npy < ave)] = 0
        # npy = npy - min_
        npy /= (max_ - min_)
        return npy

    def individual_analyze(self, gt_heatmap, target_heatmap):
        """
        Args:
            heatmap: numpy file path
        """
        gt = np.load(gt_heatmap)
        gt = self.__norm_for_numpy(gt)
        assert gt.shape == self.size
        
        target = np.load(target_heatmap)
        target = self.__norm_for_numpy(target)
        assert target.shape == self.size


        l1_err = np.sum(np.abs(gt - target)) / (self.size[0]*self.size[1])

        logging.info(f"The heatmap l1 loss of prediction heatmap is {l1_err} per pixel")

        l2_err = np.sum(np.power(gt - target, 2)) / (self.size[0]*self.size[1])

        logging.info(f"The heatmap l2 loss of prediction heatmap is {l2_err} per pixel")

        return l1_err, l2_err

    def high_volume_analyze(self, gt_folder, target_folder):
        """
        """
        logging.info(f'Load from {gt_folder}, {target_folder}')
        gt_list = glob.glob(os.path.join(gt_folder, '*.npy'))
        gt_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))

        target_list = glob.glob(os.path.join(target_folder, '*.npy'))
        target_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))


        
        l_gt = len(gt_list)
        l_target = len(target_list)
        assert l_gt == l_target

        logging.info(f'Starting analyze load 2x{l_gt} heatmaps totally.')
        l1_tot = 0
        l2_tot = 0

        for i, gt in enumerate(gt_list):
            l1_temp, l2_temp = self.individual_analyze(os.path.join(gt_folder, gt), os.path.join(target_folder, target_list[i]))
            l1_tot += l1_temp
            l2_tot += l2_temp
        
        l1_tot /= l_gt
        l2_tot /= l_gt

        logging.info(f'l1 loss in total = {l1_tot}')
        logging.info(f'l2 loss in total = {l2_tot}')

    def __read_heatmap(self, heatmap_name):
        """
        Args:
            heatmap_path
        Returns:
            np.array - heatmap
        """
        res = np.load(heatmap_name)
        
        return res

    def __calculate_difference(self, h1, h2):
        """multiply by coordinates 
        Args:   
            h1: heatmap 1
            h2: heatmap 2
        Returns:
            np.sum(np.sum(np.abs(h1 - h2)))
        """
        # h1 = h1*(h1>0.9)
        # h2 = h2*(h2>0.9)

        res = 0
        meshs = np.meshgrid(self.norm_size, self.norm_size)

        mesh_x = np.array(meshs[0]) + 1
        mesh_y = np.array(meshs[1]) + 1


        res += np.sum(np.sum(np.abs(h1*mesh_x - h2*mesh_x))) / self.norm_size**2
        # res += np.sum(np.sum(np.abs(h1*mesh_y - h2*mesh_y))) #/ self.norm_size**2

        return res
    
    def accuracy_calculate(self, h1_folder, h2_folder, mod='ori-dist'):
        """
        Workflow:
            1. read heatmaps
            2. cycle sub
            3. avarage
        """

        logging.info(f'load heatmaps from {h1_folder}  & {h2_folder}')
        h1_list = glob.glob(os.path.join(h1_folder, '*.npy'))
        h1_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))

        h2_list = glob.glob(os.path.join(h2_folder, '*.npy'))
        h2_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))

        l = len(h1_list)
        assert len(h1_list) == len(h2_list), f'Different length: {l}!'

        err = 0
        max_err = 0
        min_err = np.Inf
        for i, h1 in enumerate(h1_list):
            # print(h1)
            h1_data = self.__read_heatmap(h1)
            h2_data = self.__read_heatmap(h2_list[i])
            err_temp = self.__calculate_difference(h1_data, h2_data)
            if err_temp > max_err:
                max_err = err_temp
            if err_temp < min_err:
                min_err = err_temp

            err += err_temp

        err /= l

        logging.info(f'The average error in {mod} is {err}')
        logging.info(f'The maximal error in {mod} is {max_err}')
        logging.info(f'The minimal error in {mod} is {min_err}')

        return err
        
if __name__ == '__main__':
    pass