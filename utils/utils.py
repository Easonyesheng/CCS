"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""

import os
import numpy as np
import random
# import torch
from pathlib import Path
import datetime
import datetime
from collections import OrderedDict
# import torch.nn.functional as F
# import torch.nn as nn
import logging
import cv2
###### check
# from utils.nms_pytorch import box_nms as box_nms_retinaNet
# from utils.d2s import DepthToSpace, SpaceToDepth

# 
#=========================================================================================================================================================@Eason
def after_cv_imshow():
    """name

        close all the show window if press 'esc'
        set after cv2.imshow()

    Args:

    Returns:

    """
    import cv2
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def img_show(img, name):
    """img show function
    """
    cv2.startWindowThread()
    img = img / (np.max(img)+1e-5)
    # img = img*255
    cv2.imshow(name+str(random.randint(0, 10000)), img)
    after_cv_imshow()

def test_dir_if_not_create(path):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:
    """
    if os.path.isdir(path):
        return True
    else:
        print('Create New Folder:', path)
        os.makedirs(path)
        return True

def npy2img_show(npyfile):
    """
    """
    img_np = np.load(npyfile)
    img_show(img_np, 'np')
    return img_np


def log_init(logfilename):
    """name

        save as 'path/name.jpg'

    Args:

    Returns:
    """
    # logging.basicConfig(filename=logfilename, level=logging.INFO)
    # logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #                 filename=logfilename,
    #                 level=logging.DEBUG)

    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    # file = open(logfilename, 'w')
    # file.close()

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(logfilename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def gray2heat_img(gray_img):
    """
    """
    gray_img = 1 - gray_img
    norm_img = np.zeros(gray_img.shape)
    norm_img = cv2.normalize(gray_img , None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)

    # img_show(norm_img, 'before')
    # gray_img = np.asarray(gray_img, dtype=np.uint8)

    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

    # cv2.startWindowThread()

    # cv2.imshow('heat', heat_img)
    # after_cv_imshow()

    return heat_img

def distort_correction_visionization(parameters, img, corner_size = 480, model='poly', order=3):
    """
    Args:
        parameters: N*1 numpy
        img: corner_size numpy
    Returns:
        img_mat corner_size numpy
    """
    import math
    assert (img.shape[0] == img.shape[1]) and (img.shape[0] == corner_size), f'img size is {img.shape}'

    img_mat = np.zeros(tuple(img.shape), dtype=np.uint8) # np.shape = [height, weight, depth] or [row, colum]
    H, W = img.shape
    dis_center_x = 0.5
    dis_center_y = 0.5

    print('Start interpolation...')
    
    for x in range(W-2):
        for y in range(H-2):
            x_nom = x / W
            y_nom = y / H

            if model == 'poly':
                x_d = 0
                y_d = 0
                index = 0
                y_num = parameters.shape[0] // 2
                for j in range(order+1):
                    for i in range(0, order-j+1):
                        x_d += parameters[index] * x**i * y**(order-j-i)
                        y_d += parameters[index+y_num] * x**i * y**(order-j-i)
                        index += 1

            if model == 'radi':
                r = (x_nom-dis_center_x)*(x_nom-dis_center_x) + (y_nom-dis_center_y)*(y_nom-dis_center_y)
                r = r**(0.5)
                coff = 0
                # print(r)
                #==================for oneP else 0
                if order == 1:
                    para = float(parameters)
                    if para > 0:
                        # para += 2
                        pass
                    coff = 1 + para * (r**2) 
                else:
                    assert order == parameters.shape[0], f'order is {order} while the parmeters has {parameters.shape[0]}'
                    for i in range(parameters.shape[0]):
                        coff += parameters[i] * r**i

                x_d = coff*(x-W/2) + W/2
                y_d = coff*(y-H/2) + H/2
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