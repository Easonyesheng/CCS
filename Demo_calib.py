''' A Demo to Perform Calibration through Our Camera Calibration Framework
Note:
    * Default chessborad square length is 1
        If your chessboard square length is a mm, you should multiply the focal length by a -> real world focal length.

    * Default chesboard image is 480x480

Copyright (c) 2022 by EasonZhang, All Rights Reserved. 
'''

from logging import root
from math import fabs
import os

from torch import FloatStorage
from utils.utils import test_dir_if_not_create
import numpy as np
import cv2

# 
from settings.settings import *
from utils import *

# basic models
from models.GetSubPixel import GetSubPixels
from models.OpenCVCalibGivenPoints import CalibGivenPoints


# high level models
from rectify import Rect
from detect import Detect
from calibration import Calib

# Flags
DistortionCorrectionFlag = False
corner_dect_flag = False


# path 
# NEED TO FOLLOW THE FIX FILE FOLDER CONFIGURATION IN README.md
root_path = 'path/to/data/folder'
# root_path = './demo_data/demo_noise' # example
dist_corr_model_path = 'path/to/distortion/correction/model/.pth'
corner_dect_model_path = 'path/to/corner/detection/model/.pth'


#-------------------------------------Distortion Correction
if DistortionCorrectionFlag:
    dist_corr = Rect('Demo')
    dist_corr.net_config['order'] = 3
    dist_corr.checkpoints_path = dist_corr_model_path
    dist_corr.save_path = root_path
    dist_corr.read_dist_imgs()
    dist_corr.predict_para()
    dist_corr.Rectify_imgs()


#------------------------------------Corner Detection
if corner_dect_flag:
    corner_dect = Detect('Demo')

    if DistortionCorrectionFlag:
        img_path = os.path.join(root_path, 'rect_img')
    else:
        img_path = os.path.join(root_path, 'img') 

    corner_dect.save_path = os.path.join(root_path, 'DetectRes')
    corner_dect.checkpoint_path = corner_dect_model_path

    test_dir_if_not_create(corner_dect.save_path)
    test_dir_if_not_create(os.path.join(corner_dect.save_path, 'color_img'))
    test_dir_if_not_create(os.path.join(corner_dect.save_path, 'heatmap'))

    corner_dect.high_volume_detect(img_path, save_flag=True)


#-----------------------------------Calibration with synthetic dataset
# heatmap == distribution map are saved in /root/DetectRes/heatmap
heatmap_path = os.path.join(root_path, 'DetectRes', 'heatmap')
ref_corner_gt_path = os.path.join(root_path, 'GT')
gt_path = os.path.join(root_path,'GT')
img_path = os.path.join(root_path, 'img')


calibrator = Calib([480,480],'Demo')
heatmap_list = calibrator.get_all_heatmaps(heatmap_path)
ret, mtx, dist, rvecs, tvecs = calibrator.calib(heatmap_list, img_path=img_path, ref_path=ref_corner_gt_path, sort_mod='gt')


print(f'The reprojection error={ret}')
print(f'The intrinsic matrix: {mtx}')

calibrator.show_accuracy_by_gt(gt_path,mtx)

