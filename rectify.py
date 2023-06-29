'''A class to analyze the radial distortion correction result'''
import skimage.measure
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import logging 
import os
import time
import glob
from tqdm import tqdm
import math
import torchvision.transforms as transforms
import re
import cv2


from utils.utils import log_init, test_dir_if_not_create
from utils.utils import img_show, distort_correction_visionization
from settings.settings import *
from models.half_unet import UNet_half_4_dc

class Rect(object):
    """
    Flows:
       s1: read ori image
       s2: distorting image
       s3: predict parameter by net
       s4: rectify
       s5: calculate SSIM & PSNR
    """

    def __init__(self, log_name):
        """
        """
        self.log_name = log_name
        self.logfile = 'AnalysisLog-'+self.log_name+'.txt'
        log_init(os.path.join(LOGFILEPATH,self.logfile))

        self.save_path = r''#os.path.join(RESSAVEPATH, 'Analysis')

        self.ori_img_path = r''
        self.dist_img_path = os.path.join(self.save_path, 'dist_img')
        test_dir_if_not_create(self.dist_img_path)
        self.dist_folder_name = 'detect_dist'
        self.rec_img_path = os.path.join(self.save_path, 'rect_img')
        test_dir_if_not_create(self.rec_img_path)
        self.rec_folder_name = 'rect_img'


        self.checkpoints_path = r''

        self.net_config = {
            'mod': 'UNet_half',
            'dist_mod': 'radi',
            'order': 3,
            'norm_size': 480
        }

        self.dist_imgs = None
        self.ori_imgs = None
        self.paras = None
        self.rect_imgs = []
        self.checkerboard_size = [7,6]
        self.name_list = []

    def read_ori_imgs(self):
        """
        """
        self.ori_imgs = []
        ori_img_list = glob.glob(os.path.join(self.ori_img_path, '*.jpg'))
        ori_img_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))
        print(ori_img_list[:6])
        # ori_img_list.sort()

        logging.info(f'Load {len(ori_img_list)} imgs from {self.ori_img_path} as ori img.')

        for ori_img in ori_img_list:
            img = Image.open(ori_img)
            img = img.convert('L')
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            img = np.array(img)
            self.ori_imgs.append(img) # saved as np

    def read_dist_imgs(self):
        """
        """
        self.dist_imgs = []
        dist_img_list = glob.glob(os.path.join(self.dist_img_path, '*.jpg'))
        dist_img_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))

        if len(dist_img_list) > 0:
            flag = True
        else:
            return False

        logging.info(f'Load {len(dist_img_list)} imgs from {self.dist_img_path} as dist img.')

        for dist_img in dist_img_list:
            name = dist_img.split('\\')[-1]
            self.name_list.append(name)
            img = Image.open(dist_img)
            img = img.convert('L')
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            img = np.array(img)
            self.dist_imgs.append(img) # saved as np
        
        return flag

    def read_rect_imgs(self):
        """
        """
        self.rect_imgs = []
        rec_img_list = glob.glob(os.path.join(self.rec_img_path, '*.jpg'))
        rec_img_list.sort(key=lambda i: int(re.search(r'(\d+)', i).group()))

        if len(rec_img_list) > 0:
            flag = True
        else:
            return False

        logging.info(f'Load {len(rec_img_list)} imgs from {self.rec_img_path} as rec img.')

        for rec_img in rec_img_list:
            img = Image.open(rec_img)
            img = img.convert('L')
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            img = np.array(img)
            self.rect_imgs.append(img) # saved as np
        
        return flag
        
    def Distort_imgs(self, mod='OnePara', save_flag=True):
        """Distort image function
        """
        assert len(self.ori_imgs) > 0, f'No ori images!'

        self.dist_imgs = []

        if mod == 'OnePara':
            k = self.__get_rand_dist_para_OnePara()

            logging.info('Starting distort images...')

            for i, img in enumerate(self.ori_imgs):
                k_temp = self.__get_rand_dist_para_OnePara()
                logging.info(f'the {i}-th image and the distortion parameter={k_temp}')
                img_dist = self.__dist_img(mod, k_temp, img)
                if save_flag:
                    img = Image.fromarray(img_dist)
                    img = img.convert('L')
                    img_path = os.path.join(os.path.join(self.save_path, self.dist_folder_name), str(i)+'.jpg')
                    img.save(img_path)
                    logging.info(f'save the distorted image as {img_path}')
                self.dist_imgs.append(img_dist)

        logging.info(f'Distortion Done, {i} images distorted totally in mod: {mod}')

    def load_dist_imgs(self):
        """Repeat
        """
        self.dist_imgs = []
        dist_img_list = glob.glob(os.path.join(self.dist_img_path, '*.jpg'))
        for dist_img_name in dist_img_list:
            dist_img_temp = Image.open(dist_img_list)
            dist_img_temp = dist_img_temp.convert('L')
            dist_img_temp = np.array(dist_img_temp)
            self.dist_imgs.append(dist_img_temp)

    def __dist_img(self, mod, k, img):
        """
        """
        img = np.array(img)

        assert img.shape[0] == 480, f'only for 480x480'

        W = 480
        H = 480

        img_mat = np.zeros((480,480))
        dis_center_x = 240
        dis_center_y = 240

        for x in range(W-2):
            for y in range(H-2):
                x_nom = (x - dis_center_x)/ (W)
                y_nom = (y - dis_center_y)/ (W)
                r = (x_nom)*(x_nom) + (y_nom)*(y_nom)
                
                if mod == 'OnePara':
                    coff = (1+k*r) # corr old
                else:
                    assert k.shape[0] > 1, f'k is {k}, while model is {mod}'
                    r = r**(0.5)

                    for i in range(k.shape[0]):
                        coff += k[i] * r**i
                
                x_d = ((coff)*(x-W/2) + W/2)
                y_d = ((coff)*(y-H/2) + H/2)
                # print(x_d, y_d)
                if x_d >= W-1 or y_d >= H-1 or x_d<0 or y_d <0:
                    # last row or colume -- ignore!
                    continue

                # bib interploration
                x_d_int = math.floor(x_d)
                y_d_int = math.floor(y_d)

                dx = x_d - x_d_int
                dy = y_d - y_d_int

                img_mat[y_d_int,x_d_int] = (1-dx)*(1-dy)*img[y,x] + dx*(1-dy)*img[y,x+1] + \
                                        (1-dx)*dy*img[y+1,x] + dx*dy*img[y+1,x+1]
                img_mat[y_d_int,x_d_int+1] = (1-dx)*(1-dy)*img[y,x+1] + dx*(1-dy)*img[y,x+2] + \
                                        (1-dx)*dy*img[y+1,x+1] + dx*dy*img[y+1,x+2]
                img_mat[y_d_int+1,x_d_int] = (1-dx)*(1-dy)*img[y+1,x] + dx*(1-dy)*img[y+1,x+1] + \
                                        (1-dx)*dy*img[y+2,x] + dx*dy*img[y+2,x+1]
                img_mat[y_d_int+1,x_d_int+1] = (1-dx)*(1-dy)*img[y+1,x+1] + dx*(1-dy)*img[y+1,x+2] + \
                                        (1-dx)*dy*img[y+2,x+1] + dx*dy*img[y+2,x+2]

        return img_mat

    def __get_rand_dist_para_OnePara(self):
        """
        """
        k_res = np.random.rand() - 0.5
        k_res *= 2
        # k_res = abs(k_res)
        return k_res

    def __preprocess_imgs_for_pre(self):
        """
        """
        self.net_input = []

        logging.info(f'Starting preprocess images...')

        for i, img in enumerate(self.dist_imgs):
            assert img.shape[0] == 480, f'the {i}-th img size is wrong, size is {img.shape}'
            assert img.shape[1] == 480, f'the {i}-th img size is wrong, size is {img.shape}'

            if len(img.shape) == 2:
                img_nd = np.expand_dims(img, axis=2)

            img_trans = img_nd

            if img_trans.max() > 1:
                img_trans = img_trans / 255

            transform_toTensor = transforms.ToTensor()

            img = transform_toTensor(img_trans)
            img = img.unsqueeze(0)
            self.net_input.append(img)

        logging.info(f'Preprocessing images done.')

    def predict_para(self):
        """
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert len(self.dist_imgs)>0, f'No distortion images'

        mod = self.net_config['mod']
        dist_mod = self.net_config['dist_mod']
        order = self.net_config['order']
        size = self.net_config['norm_size']

        logging.info(f'''Starting load model: 
            model name:       {mod}
            distortion model: {dist_mod}
            order:            {order}
            image size:             {size}
        ''')

        net = UNet_half_4_dc(order, device, 1,1,True, size, dist_mod)

        net.eval()

        checkpoints = self.checkpoints_path

        net.to(device=device)

        sta = torch.load(checkpoints)
        net.load_state_dict(sta)
        logging.info(f'load model from {checkpoints}')
        
        self.paras = []

        self.__preprocess_imgs_for_pre()
        
        img_input_num = len(self.net_input)
        logging.info(f'Preprocess {img_input_num} images.')

        for i, img in enumerate(self.net_input):
            img = img.to(device, dtype=torch.float32)

            
            with torch.no_grad():
                start = time.time()
                net_out = net(img)
                end = time.time()
                para = net_out['parameters'].detach().cpu().numpy()
                para = np.squeeze(para)
            

            cost_time = end-start

            logging.info(f'predict the {i}-th image in time:{cost_time}s')
            logging.info(f'the parameters are: {para}')

            self.paras.append(para)
        
        para_num = len(self.paras)
        logging.info(f'Prediction is done. Got {para_num} parameters totally.')
        
    def Rectify_imgs(self, save_flag=True):
        """
        """
        self.rect_imgs = []

        assert len(self.paras) == len(self.dist_imgs), f'Have {self.paras} parameters, while {self.dist_imgs} diostortion images'

        logging.info(f'Starting rectify images...')

        for i, img in enumerate(self.dist_imgs):
            img_rect = distort_correction_visionization(self.paras[i], img, model=self.net_config['dist_mod'], order=self.net_config['order'])

            self.rect_imgs.append(img_rect) # saved as np

            img_rect = Image.fromarray(img_rect)
            img_rect = img_rect.convert('L')
            
            logging.info(f"Rectified {i}-th image")

            if save_flag:
                img_save_path = os.path.join(self.rec_img_path, self.name_list[i])
                logging.info(f'Saved in {img_save_path}')
                img_rect.save(img_save_path)

        logging.info(f'Rectify done.')

    def analyse_script(self):
        """
        """
        self.read_ori_imgs()

        # dist_flag = self.read_dist_imgs()

        # if not dist_flag:
        self.Distort_imgs()

        self.predict_para()

        # rec_flag = self.read_rect_imgs()

        # if not rec_flag:
        self.Rectify_imgs()

        #=========================
        ssim_tot = 0
        psnr_tot = 0
        for i, img in enumerate(self.rect_imgs):
            ssim = self.cal_SSIM(self.ori_imgs[i], img)
            psnr = self.cal_PSNR(self.ori_imgs[i], img)
            logging.info(f'The {i}-th img: PSNR={psnr}, SSIM={ssim}')
            ssim_tot += ssim
            psnr_tot += psnr
        
        ssim_tot /= i
        psnr_tot /= i

        logging.info(f'The average PSNR={psnr_tot}')
        logging.info(f'The average SSIM={ssim_tot}')

    def cal_SSIM(self, ori_img, rec_img):
        """
        """
        ssim = skimage.measure.compare_ssim(ori_img, rec_img, data_range=255)

        return ssim

    def cal_PSNR(self, ori_img, rec_img):
        """
        """
        psnr = skimage.measure.compare_psnr(ori_img, rec_img, 255)

        return psnr

    def __detect_corners_opencv(self, img):
        """
        """
        # img = img.convert('L')
        # img = img.filter(ImageFilter.GaussianBlur(radius=2))
        # img = cv2.resize(img, tuple(self.img_size))
        img.astype(np.int8)
        
        flag = True
        corners2 = []
        # img_show(img, 'ori')
        # The order of chess board size decide the corner order: first is the first.
        ret, corners = cv2.findChessboardCorners(img, (self.checkerboard_size[1]-1,self.checkerboard_size[0]-1),None)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if ret :
            corners2 = cv2.cornerSubPix(img,corners,(10,10), (-1,-1), criteria)
            self.opencv_corner = corners2
            # self.__draw_corner(img, corners2)
        
            # assert len(corners2) == (self.checkboard_size[0]-2)*(self.checkboard_size[1]-1)
        else:
            logging.warning('Original Image Corner Cannot completely Find')
            flag = False
        # print(corners2)
       
        return flag, corners2

    def detect_corner_opencv_and_analyse(self):
        """workflow
            detect corners of ori image -> ori_detected_corner
            detect corners of dist image -> dist_detected_corner
            detect corners of rect image -> rect_detected_corner
            err_ori_dist
            err_ori_rect
        """
        ori_corners = []
        dist_corners = []
        rect_corners = []
        err_ori_dist = 0
        err_ori_rect = 0
        

        self.read_dist_imgs()
        self.read_ori_imgs()
        self.read_rect_imgs()

        assert len(self.ori_imgs) > 0
        assert len(self.ori_imgs) == len(self.dist_imgs)
        assert len(self.ori_imgs) == len(self.rect_imgs)

        count_o_d = 0
        count_o_r = 0
        for i in range(len(self.ori_imgs)):
            flag_ori , temp_cor_ori = self.__detect_corners_opencv(self.ori_imgs[i])
            flag_dist , temp_cor_dist = self.__detect_corners_opencv(self.dist_imgs[i])
            flag_rect , temp_cor_rect = self.__detect_corners_opencv(self.rect_imgs[i])

            if not flag_ori:
                print(f'ori {i} img cannot detect')
            if not flag_dist:
                print(f'dist {i} img cannot detect')
            if not flag_rect:
                print(f'rect {i} img cannot detect')

            if flag_ori and flag_dist:
                cor_o = np.array(temp_cor_ori)
                cor_d = np.array(temp_cor_dist)
                err_t1 = np.sum(np.sum(np.abs(cor_o-cor_d))) / cor_o.shape[0]
                err_ori_dist += err_t1
                count_o_d += 1

            if flag_ori and flag_rect:
                cor_o = np.array(temp_cor_ori)
                cor_r = np.array(temp_cor_rect)
                err_t2 = np.sum(np.sum(np.abs(cor_o-cor_r))) / cor_o.shape[0]
                err_ori_rect += err_t2
                count_o_r += 1

        err_ori_dist /= count_o_d
        err_ori_rect /= count_o_r

        print(f'ori-dist = {err_ori_dist} \nori-rect={err_ori_rect}')
if __name__ == '__main__':
    

    #===================================for dist calib
    test = Rect('calib_dist')
    test.net_config['order'] = 3
    test.checkpoints_path = r''
    test.save_path = r''
    test.dist_img_path = r''

    test.read_dist_imgs()
    test.predict_para()
    test.Rectify_imgs()
