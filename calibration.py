''' A Class to Perform Calibration '''

from heapq import nsmallest
import os
from traceback import print_tb
import numpy as np
import glob
import logging
import random
import matplotlib.pyplot as plt
import sys

from models.GetSubPixel import GetSubPixels
from models.OpenCVCalibGivenPoints import CalibGivenPoints
from settings.settings import *
from utils.utils import *

class Calib(object):
    """Perform Calibration
    """
    def __init__(self, img_size, log_name= ''):
        """
        Args;
            img_size = [x_size, y_size]
        """
        self.subpixel_exactor = GetSubPixels(save_path='',show_flag=False)

        self.chessboardsize = chessboardsize
        self.calibtor = CalibGivenPoints(chessboardsize, '', 40)

        self.size = 480
        self.img_size = img_size
        self.x_ratio = self.size / img_size[0]
        self.y_ratio = self.size / img_size[1]
        self.corner_num = corner_num
        self.img_path = r''
        self.log_file_path = os.path.join(LOGFILEPATH, 'calib_'+log_name+'.txt')
        log_init(self.log_file_path)

    def load_heatmaps(self, heatmap_list):
        """load imgs by list
        """
        img_num = len(heatmap_list)
        heatmaps = []
        heatmap_names = []

        for heatmap_name in heatmap_list:
            heatmap_temp = np.load(heatmap_name)
            # name_temp = heatmap_name.split('/')[-1] # mac
            name_temp = heatmap_name.split('\\')[-1] # win
            heatmaps.append(heatmap_temp)
            heatmap_names.append(name_temp)
        
        return heatmaps, heatmap_names
    
    def get_coor_from_heatmap(self, heatmap, name, sort_mod='gt', ORT=20, ref_path='', img_path='', show_flag=False):
        """For a single heatmap
        Args:
            name: index.npy
        Returns:
            cs: ndarray 30x2
        """
        corner_flag = False
        self.subpixel_exactor.load_distribution_map(name, heatmap, self.corner_num, chessboardsize)
        flag, cand = self.subpixel_exactor.thredshold_locate()
        if not flag:
            cs, s_err = self.subpixel_exactor.sub_pixel_localization(cand)
            if s_err > ORT:
                pass
            else:
                gt = os.path.join(ref_path, name)
                self.subpixel_exactor.load_gt_4_calibsort(gt)
                if sort_mod == 'gt':
                    cs = self.subpixel_exactor.sort_by_gt_data(show_flag=False)
                    cs = self.subpixel_exactor.collineation_refinement(cs)
                    err = self.subpixel_exactor.cal_err()
                    print('corner error=', err)
                    corner_flag = True
                elif sort_mod == 'corner':
                    img_name = os.path.join(img_path, name.split('.')[0]+".jpg")
                    img = cv2.imread(img_name)
                    self.subpixel_exactor.opencv_find_corner(img)
                    cs = self.subpixel_exactor.sort_by_corners()
                    cs = self.subpixel_exactor.collineation_refinement(cs)
                    err = self.subpixel_exactor.cal_err()
                    print('corner error=', err)
                    corner_flag = True
                elif sort_mod == 'OR':
                    img_name = os.path.join(img_path, name.split('.')[0]+".jpg")
                    img = cv2.imread(img_name)
                    flag, cs = self.subpixel_exactor.run_OR(heatmap, img)
                    err = self.subpixel_exactor.cal_err()
                    print('corner error=', err)
                    corner_flag = flag
                cs = np.array(cs)
        else:
            return corner_flag, None

        return corner_flag, cs
 

    def get_coors_from_heatmap_list(self, heatmap_list, sort_mod='gt', ORT=20, ref_path='', img_path=''):
        """
        Returns:
            names: [index.npy, ...]
        """
        pose_num = len(heatmap_list)
        corners = []
        names = []
        
        for i in range(pose_num):
            heatmap = np.load(heatmap_list[i])
            name = heatmap_list[i].split('\\')[-1]
            corner_flag, cs = self.get_coor_from_heatmap(heatmap, name, sort_mod=sort_mod, ORT=ORT, ref_path=ref_path, img_path=img_path)
            if corner_flag:
                corners.append(cs.astype('float32'))
                names.append(name)
        
        return corners, names
        
    def calib(self, heatmap_list, sort_mod='gt', ORT=20, img_path='', ref_path='',fix_CM_flag=False, f=0, p=0):
        """calib with a set of heatmaps
        """
        img_num = len(heatmap_list)

        heatmaps, heatmap_names = self.load_heatmaps(heatmap_list)
        corners = []
        corner_names = []
        print('Get sub-pixel corners...')
        for i, heatmap in enumerate(heatmaps):
            flag, cs = self.get_coor_from_heatmap(heatmap, heatmap_names[i], sort_mod, ORT, ref_path, img_path)
            if flag:
                corners.append(cs.astype('float32'))
                corner_names.append(heatmap_names[i])
            else:
                img_num -= 1
        self.calibtor.img_number = img_num
        self.calibtor.get_o_points()
        print(f'Calibration with {img_num} images.')
        self.calibtor.img_points = corners
        if fix_CM_flag:
            ret, mtx, dist, rvecs, tvecs = self.calibtor.calib_with_fix_IM(f,p)
            
        else:
            ret, mtx, dist, rvecs, tvecs = self.calibtor.calibrate()
        
        return ret, mtx, dist, rvecs, tvecs

    def calib_by_subpixel_res(self,subpixel_list, fix_flag=False, fx=0, fy=0, px=0, py=0):
        """
        """
        corners = []
        pose_num = len(subpixel_list)
        for subpixel in subpixel_list:
            cs_t = np.load(subpixel)
            corners.append(cs_t.astype('float32'))  

        self.calibtor.img_number = pose_num
        self.calibtor.get_o_points()
        print(f'Calibration with {pose_num} images.')
        self.calibtor.img_points = corners
        if fix_flag:
            ret, mtx, dist, rvecs, tvecs = self.calibtor.calib_with_fix_IM(fx,fy,px,py)
        else:
            ret, mtx, dist, rvecs, tvecs = self.calibtor.calibrate()
        
        return ret, mtx, dist, rvecs, tvecs

    def save_subpixel_corners(self, heatmap_path, sort_mod='gt', ORT=20, ref_path='', img_path='', save_path=r''):
        """save subpixel coordinates
        """
        heatmap_list = self.get_all_heatmaps(heatmap_path)
        corners, names = self.get_coors_from_heatmap_list(heatmap_list, sort_mod, ORT, ref_path, img_path)
        for i, corner in enumerate(corners):
            np.save(os.path.join(save_path, names[i]), corner)
        
        print(f'save subpixel corners in {save_path}')

    def calib_by_RANSAC_practical(self, heatmap_folder, subpixel_path, max_iter_num=100, least_pose_num=20, outlier_threshold=0.8, inlier_threshold=2/3, sort_mod='gt', ORT=20, ref_path=r'', ref_mod='gt',save_flag=False, draw_flag=False):
        """
        Args:
            least_pose_num : use <=* images to calib every time
            outlioutlier_threshold : if RPE > *, set it as outlier
            inlier_threshold: if inliers numbers > (* x img_num) -> stop calib 
        """
        if save_flag:
            self.save_subpixel_corners(heatmap_folder, sort_mod,ORT,ref_path,save_path=subpixel_path)
        
        heatmap_list = glob.glob(os.path.join(heatmap_folder,'*.npy'))
        subpixel_list = glob.glob(os.path.join(subpixel_path,'*.npy'))
        total_num = len(heatmap_list)
        assert total_num > least_pose_num, 'Images Not Enough!'

        inlier_num = 0
        inlier_max = 0
        best_models = [] 
        inlier_max_ip_error = 0
        iter_count = 0
        IPs = []
        I_nums = []
        inlier_num_threshold = int(inlier_threshold * total_num)

        while inlier_num < inlier_num_threshold and iter_count < max_iter_num:
            # get poses
            pose_list = self.__choose_pose_randomly(subpixel_path, least_pose_num)

            # calib to get K
            ret, mtx, _, _, _ = self.calib_by_subpixel_res(pose_list)
            fx = mtx[0,0]
            fy = mtx[1,1]
            px = mtx[0,2]
            py = mtx[1,2]

            # calib all fix K get Rs,ts
            print('Counting inliers...')
            _, _, dist, rvecs, tvecs = self.calib_by_subpixel_res(subpixel_list,fix_flag=True,fx=fx,fy=fy,px=px,py=py)

            # calculate RPEs
            RPEs = self.calibtor.call_re_projection_errors(rvecs,tvecs,mtx,dist)

            # count inlier numbers
            count = 0
            for RPE in RPEs:
                # print(RPE)
                if RPE < outlier_threshold:
                    count+=1

            # change vars
            inlier_num = count
            if inlier_num > inlier_max:
                inlier_max = inlier_num
                # inlier_max_ip_error = e_ip
                best_models = []
                best_models.append(mtx)
            elif inlier_num == inlier_max:
                best_models.append(mtx)
            iter_count+=1

        return best_models
    
    def calib_RANSAC_OpenCV(self,img_folder, max_iter_num=3, least_pose_num=20, outlier_threshold=0.1, inlier_threshold=2/3):
        """
            You are recommended to use large max_iter_num to acquire better model.
        Args:
            least_pose_num : use <= ([]ximages) to calib every time
            outlioutlier_threshold : if RPE > [], set it as outlier
            inlier_threshold: if inliers numbers > ([] x img_num) -> stop calib 
        Returns:
            a list of Intrinsic parameters with max Inlier number
            You should choose one of them and NOT choose the one much different from others.
        """
        
        img_list = glob.glob(os.path.join(img_folder,'*.jpg'))
        total_num = len(img_list)
        assert total_num > least_pose_num, 'Images Not Enough!'

        inlier_num = 0
        inlier_max = 0
        best_models = []
        inlier_max_ip_error = 0
        iter_count = 0
        IPs = []
        I_nums = []
        inlier_num_threshold = int(inlier_threshold * total_num)

        while inlier_num < inlier_num_threshold and iter_count < max_iter_num:
            # get poses
            img_list_per_calib = self.__choose_img_randomly(img_list, least_pose_num)

            # calib to get K
            self.calibtor.load_imglist(img_list_per_calib)
            self.calibtor.get_points_by_images()
            ret, mtx, _, _, _ = self.calibtor.calibrate()
            fx = mtx[0,0]
            fy = mtx[1,1]
            px = mtx[0,2]
            py = mtx[1,2]

            # calib all fix K get Rs,ts
            print('Counting inliers...')
            self.calibtor.load_imglist(img_list)
            self.calibtor.get_points_by_images()
            _, _, dist, rvecs, tvecs = self.calibtor.calib_with_fix_IM(fx=fx,fy=fy,px=px,py=py)

            # calculate RPEs
            RPEs = self.calibtor.call_re_projection_errors(rvecs,tvecs,mtx,dist)

            # count inlier numbers
            count = 0
            for RPE in RPEs:
                # print(RPE)
                if RPE < outlier_threshold:
                    count+=1

            # print(f'{count} inliers with IP error = {e_ip}')

            # change vars
            inlier_num = count
            if inlier_num > inlier_max:
                best_models = []
                inlier_max = inlier_num
                # inlier_max_ip_error = e_ip
                best_models.append(mtx)
            elif inlier_num == inlier_max:
                best_models.append(mtx)
            iter_count+=1
            print(f'{iter_count} iter with max {inlier_max} inliers')

        return best_models
         
    def __call_inliers_num(self, RPEs, threshold):
        """
        """
        count = 0
        for rpe in RPEs:
            if rpe > threshold:
                count+=1
        return count

    def draw_error_distribution(self, errs_f, errs_p):
        """
        """
        plt.plot(errs_f, errs_p, 'ro')
        plt.show()

    def draw_ret_and_err_ip_distribution(self, errs_ip, rets):
        """
        """
        plt.plot(errs_ip, rets, 'ro')
        plt.xlabel('Intrinsic parameters error')
        plt.ylabel('Reprojection error')
        plt.show()

    def __choose_img_randomly(self, img_list, max_num):
        """get a name list of files (subpixel corners) in a given folder
        Args:
            folder: file folder path
        """
        name_list = []
        pose_num = random.randint(3,max_num)
        # all_imgs= glob.glob(os.path.join(folder, '*.jpg'))


        for i in range(pose_num):
            pose_index = random.randint(0,pose_num-1)
            name_temp = img_list[pose_index]
            if name_temp not in name_list:
                name_list.append(name_temp)
        
        return name_list

    def __choose_pose_randomly(self, folder, max_num):
        """get a name list of files (subpixel corners) in a given folder
        Args:
            folder: file folder path
        """
        name_list = []
        pose_num = random.randint(3,max_num)
        all_subpxiels = glob.glob(os.path.join(folder, '*.npy'))


        while(len(name_list) < pose_num):
            pose_index = random.randint(0,pose_num-1)
            name_temp = all_subpxiels[pose_index]
            if name_temp not in name_list:
                name_list.append(name_temp)
        
        return name_list
    
    def get_all_heatmaps(self, folder):
        """
        Args:
            folder : /root/DetectRes/heatmap
        """
        heatmap_list = glob.glob(os.path.join(folder, '*.npy'))

        return heatmap_list
    
    def show_accuracy_by_gt(self, gt_path=r'', mtx=None):
        """
        """
        gt_list = glob.glob(os.path.join(gt_path,'*.npy'))
        gt_temp = np.load(gt_list[0],allow_pickle=True)
        gt_temp = gt_temp.item()
        K_gt = gt_temp['K']
        err_f, err_p, err_ip = self.cal_accuracy_by_gt(gt_path,mtx)
        print(f'The ground truth intrinsic matrix is {K_gt}')
        print(f'The focal length error = {err_f}')
        print(f'The principle points error = {err_p}')
        print(f'The intrinsic parameters error = {err_ip}')


        return err_f, err_p, err_ip

    def cal_accuracy_by_ref_K(self, ref_K, mtx):
        """
        """
        K_gt = ref_K
        fx_gt = K_gt[0,0]
        fy_gt = K_gt[1,1]
        px_gt = K_gt[0,2]
        py_gt = K_gt[1,2]

        fx_pred = mtx[0,0]
        fy_pred = mtx[1,1]
        px_pred = mtx[0,2]
        py_pred = mtx[1,2]

        err_f = (abs(fx_pred-fx_gt)+abs(fy_pred- fy_gt))/2
        err_p = (abs(px_pred-px_gt)+abs(py_pred- py_gt))/2
        err_ip = (err_f+err_p)/2

        return err_f, err_p, err_ip

    def show_accuracy_by_ref_K(self,ref_K, mtx):
        err_f, err_p, err_ip = self.cal_accuracy_by_ref_K(ref_K, mtx)
        print(f'The ground truth intrinsic matrix is {ref_K}')
        print(f'The focal length error = {err_f}')
        print(f'The principle points error = {err_p}')
        print(f'The intrinsic parameters error = {err_ip}')
    
    def cal_accuracy_by_gt(self, gt_path=r'', mtx=None):

        gt_list = glob.glob(os.path.join(gt_path,'*.npy'))
        gt_temp = np.load(gt_list[0],allow_pickle=True)
        gt_temp = gt_temp.item()

        K_gt = gt_temp['K']
        fx_gt = K_gt[0,0]
        fy_gt = K_gt[1,1]
        px_gt = K_gt[0,2]
        py_gt = K_gt[1,2]

        fx_pred = mtx[0,0]
        fy_pred = mtx[1,1]
        px_pred = mtx[0,2]
        py_pred = mtx[1,2]

        err_f = (abs(fx_pred-fx_gt)+abs(fy_pred- fy_gt))/2
        err_p = (abs(px_pred-px_gt)+abs(py_pred- py_gt))/2
        err_ip = (err_f+err_p)/2


        return err_f, err_p, err_ip

    def post_process(self, mtxs):
        """
        """
        mtxs_fix = []
        for mtx in mtxs:
            fx = mtx[0,0]
            fy = mtx[1,1]
            px = mtx[0,2]
            py = mtx[1,2]

            fx *= self.x_ratio
            px *= self.x_ratio
            fy *= self.y_ratio
            py *= self.y_ratio

            mtx_fix = [ [fx, 0, px], 
                        [0 ,fy, py], 
                        [0, 0,  1]]

            mtxs_fix.append(mtx_fix)

        return mtxs_fix        





if __name__ == '__main__':
    test = Calib(img_size=[480,480] ,log_name='test')
    root_path = r''
    heatmap_folder = r''
    ref_path = r''
    img_folder = r'D:\DeepCalib\CCS\Datasets\demo_iter_1\img'
    mtxs = test.calib_RANSAC_OpenCV(img_folder)
    mtxs = test.post_process(mtxs)
    print(mtxs)
    