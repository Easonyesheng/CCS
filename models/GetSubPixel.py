''' get sub-pixel points coordinates from network output in singel distribution map'''

# from json.tool import main
# from locale import D_FMT
# from operator import index
import cv2
from matplotlib.pyplot import draw
import numpy as np 
import copy
import os 
import logging
from PIL import Image, ImageDraw
from math import log, sqrt

from utils.utils import *
from settings.settings import *

class GetSubPixels(object):
    """Get sub-pixel corner coordinates of a singel distribution map
    Workflow:
        read distribution map -> locate by thredshold -> subpixel refine -> output corner
    """

    def __init__(self, save_path, show_flag=False):
        """
        
        """
        self.corner = None
        self.save_path = save_path
        self.show_flag = show_flag
        self.gt_flag = False
    
    def load_distribution_map(self, index, distribution_map, corner_num, corner_size=[5,4]):
        """
        Args:
            map: 480x480; 0~1;
        """
        self.distribution_map = distribution_map
        # logging.info(f'Current distribution map {index} with corners number {corner_num}')
        self.corner_num = corner_num
        self.index = index
        self.corner_size = corner_size

        if self.show_flag:
            img_ori = Image.fromarray(self.distribution_map*255)
            img_ori.show()
        
    def thredshold_locate(self, cluster_dist = 12**2, thredshold_height=0.01, step=0.005, max_iter=100):
        """use threshold to locate

        Returns:
            flag: rejection flag - True for rejection
            candidates: putative location

        """
        # start
        locate_number = 0
        candidates = []
        iter_count = 0
        corner_num = self.corner_num
        flag = False
        dist = cluster_dist

        while locate_number != (corner_num):
            
            # iter count
            iter_count += 1
            if iter_count > max_iter:
                break

            # height adaptive
            if locate_number < corner_num:
                thredshold_height += step
            else:
                thredshold_height -= (step/8)

            # reset
            locate_number = 0

            # get highest value
            threshold = self.distribution_map.max() - thredshold_height
            index = np.where(self.distribution_map > threshold)

            # choose candidates from index with clustering
            i = 0 # pt for index
            while i < len(index[0]):
                if len(candidates) == 0:
                    candidates.append([index[0][0], index[1][0]])
                    locate_number+=1
                else:
                    # min dist get
                    d = np.Inf
                    for can in candidates:
                        d_t = (index[0][i]-can[0])**2 + (index[1][i]-can[1])**2
                        if d_t < d:
                            d = d_t
                    
                    # clustering
                    if d > dist:
                        candidates.append([index[0][i], index[1][i]])
                        locate_number += 1
                        # print(locate_number)
                        # if self.show_flag:
                        #     corner = np.zeros((480,480))
                        #     for can in candidates:
                        #         corner[can[0],can[1]] = 255
                        #     corner = Image.fromarray(corner)
                        #     corner.show()
                
                i += 1

        if self.show_flag:
            corner = np.zeros((480,480))
            for can in candidates:
                corner[can[0],can[1]] = 255
            corner = Image.fromarray(corner)
            corner.show()

        l = len(candidates)
        if len(candidates) == corner_num:
            # logging.info(f'Well putative located of {self.index} map with {l} candicates')
            pass
        else:
            logging.info(f'Reject Image {self.index} with candidates={l}')
            flag = True
        candidates = self.sort_points_list(candidates)

        return flag, candidates
            

    def __subpixel_localization_for_single_point(self, locate, gt_variance=4.8000**2, window_size=2*4+1, Opencv_flag=False):
        """Apply Gaussian Surface Fitting Algorithm to refine sub-pixel coordinates
            The variance is used as outlier rejection
        Args:
            locate: [x, y]
            sigma_gt = 4.800000 experimentally [for the reason:?] sigma**2=23.04
            window size = odd number x -> step = x//2 -> d=[-step~step] -> points number=x**2

        Returns:
            mu: sub-pixel coordiantes-matrix [1x2]
            sigma err: 
        """
        W = 480
        points_num = window_size**2
        x, y = locate[0], locate[1]
        d = range(-1*window_size, window_size)
        xs = [x+i for i in d]
        ys = [y+i for i in d]

        points = []
        for i in range(window_size):
            for j in range(window_size):
                points.append([xs[i],ys[j]])
        
        Is = []
        for i in range(points_num):
            Is.append(self.distribution_map[points[i][0],points[i][1]])
        
        A = []
        for i in range(points_num):
            A.append(abs(Is[i])*log(abs(Is[i])+1e-5))
        A = np.matrix(A).T

        B = []
        for i in range(points_num):
            B.append([Is[i], Is[i]*points[i][0], Is[i]*points[i][1], Is[i]*points[i][0]**2,Is[i]*points[i][1]**2])
        B = np.matrix(B)

        C = B.I * A

        y_ = -1*C[2]/(2*C[4])
        # y_ = C[2]*23.04

        x_ = -1*C[1]/(2*C[3])
        # x_ = C[1]*23.04


        sigma_1 = -1/(2*C[3])
        sigma_2 = -1/(2*C[4])

        if Opencv_flag:
            mu = np.array([float(y_),float(x_)])
        else:
            mu = np.array([float(x_),float(y_)])
        sigma_err = abs(sigma_1-gt_variance) + abs(sigma_2-gt_variance)
        sigma_err /= 2

        return mu, sigma_err

    def sub_pixel_localization(self, candidates):
        """ get & save

        Returns:
            corners: Nx1x2
        """
        corners = []
        sigma_err = 0
        for can in candidates:
            cor, s_e = self.__subpixel_localization_for_single_point(can)

            # # outlier rejection
            # if s_e > 4:
            #     cor = np.array([[0,0]])

            sigma_err += s_e
            corners.append(cor)
        
        # corners = self.sort_points_list(corners)
        corners = np.array(corners)
        s = corners.shape
        sigma_err /= self.corner_num

        # np.save(os.path.join(self.save_path, self.index+'.npy'), corners)

        # logging.info(f'Get sub-pixel corners with shape {s} and sigma error = {sigma_err}')

        self.corner = corners

        return corners, sigma_err



    def collineation_refinement(self, sorted_corner):
        """fit lines -> calc the intersections
        Args:
            sorted_corner: every [h] points is a set, totally [w] sets, np.array
                point = corner[i] = [x, y]
        """
        cs = None
        w, h = chessboardsize[0], chessboardsize[1]
        corner_num = self.corner_num

        hori_lines = []
        vec_lines = []

        # horizen lines fitting
        for i in range(w):
        # for i in range(1):
            points = self.__get_points(sorted_corner, i, w, h, 'Hor', False)
            line = self.__fit_line(points, show_flag=False)
            hori_lines.append(line)
            # print(i)

        # vertical lines fitting
        for i in range(h):
        # for i in range(1):
            points = self.__get_points(sorted_corner, i, w, h, 'Vec', False)
            line = self.__fit_line(points, show_flag=False)
            vec_lines.append(line)
        
        # print(f"Get ", len(hori_lines), "hori_lines, and ", len(vec_lines), "vec_lines.")
        # calc line intersections
        # fix horizen line and calc the inter with each vec line
        cs = []
        for hline in hori_lines:
            for vline in vec_lines:
                point = self.__calc_insec(hline, vline)
                cs.append(point) 
        cs = np.array(cs)
        cs = np.squeeze(cs)
        # print(cs.shape)
        # self.show_corner_in_sort(cs)
        self.corner = cs

        return cs


    def __calc_insec(self, line1, line2):
        """
        Return:
            [x,y]
        """
        [k1, b1] = line1
        [k2, b2] = line2
        x = (b1 - b2) / (k2 - k1)
        y = k1 * x + b1

        return np.array([x, y])

    def __get_points(self, corners, index, w, h, mod='Hor', show_flag=False):
        """
        Args:
            same as collineation_refinement
            index
        Returns:
            a set of points to fit line: array([[x1,y1],...])
        """
        corners = corners.tolist()
        points = []
        if mod == 'Hor':
            points = corners[index * h : (index + 1) * h]
        elif mod == 'Vec':
            for i in range(index, len(corners), h):
                points.append(corners[i])
        else:
            print('Invalid mod in get_points')
        
        if show_flag:
            img = np.zeros((IMAGE_SIZE,IMAGE_SIZE))
            for i in points:
                img[int(i[0]), int(i[1])] = 255
            img = Image.fromarray(img)
            img.show()
        
        return points

    def __fit_line(self, points, show_flag=False):
        """
        Args:
            points: [[x1,y1],[x2,y2],...], np.adarray
        Returns:
            [k,b]: y = kx + b
        """
        points = np.array(points)
        output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k* output[2]

        if show_flag:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE,3))
            pt1 = (b, 0)
            pt2 = (240*k+b, 240)
            cv2.line(img, pt1, pt2, (0,0,255), 1, 4)
            img_show(img, 'line')
            after_cv_imshow()

        return [k,b]


    
    def show_corner_in_sort(self, corner):
        """self.sort_by_gt_data(show_flag=True)
        """
        img_corner = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        for i in range(corner.shape[0]):
            print(corner[i],'=?',self.gt[i])
            img_corner[int(corner[i][0]), int(corner[i][1])] = min(255,i*15)
        img_corner = Image.fromarray(img_corner)
        img_corner.show()

    def load_gt_corners_for_analysis(self, gt_name, info_flag=False):
        """
        """
        gt = []
        if info_flag:
            gt_info = np.load(gt_name, allow_pickle=True)
            # print(gt_info)
            gt_info = gt_info.item()
            gt_corners = gt_info['img_points']
            for cor in gt_corners:
                x = cor[1,0]
                y = cor[0,0]
                gt.append([x,y])
        else:
            gt_cor = np.load(gt_name)
            for i in range(gt_cor.shape[0]):
                gt.append([gt_cor[i,1,0],gt_cor[i,0,0]])

        gt = self.sort_points_list(gt)
        gt = np.array(gt)
        np.save(os.path.join(self.save_path,'GT_corner.npy'), gt)
        shape = gt.shape
        logging.info('gt corners with shape {shape} is loaded.')
        return gt

    def load_gt(self, gt_name):
        """ as info
        """
        # assert gt_name.split('\\')[-1].split('.')[0] == self.name
        info = np.load(gt_name, allow_pickle=True)
        info = info.item()
        gt = []
        img_points_raw = info['img_points']
        for corner in img_points_raw:
            gt.append([corner[0,0], corner[1,0]])

        self.gt = self.sort_points_list(gt)
        self.gt = np.array(self.gt)
        self.gt_flag = True
        return self.gt

    def load_opencv_corners_for_sort(self, corner_name):
        """
        """
        corner = np.load(corner_name)
        self.sort_ref_corner = []
        for i in range(corner.shape[0]):
            self.sort_ref_corner.append([corner[i][0,1],corner[i][0,0]])
            # self.sort_ref_corner.append([corner[i][0,1],corner[i][0,0]])
        
        self.sort_ref_corner = np.array(self.sort_ref_corner)

    def load_gt_4_calibsort(self, gt_name):
        """
        """
        res = np.load(gt_name, allow_pickle=True)
        # print(res)
        res = res.item()
        points = []
        for corner in res['img_points']:
            points.append([corner[0,0], corner[1,0]])

        points = np.array(points)
        self.gt = points
        self.gt_flag = True
        return self.gt

    def cal_err(self):
        self.err = 0
        temp_corner1 = copy.deepcopy(self.corner)
        temp_corner2 = copy.deepcopy(self.gt)
        temp_corner1 = self.sort_points_list(temp_corner1)
        temp_corner2 = self.sort_points_list(temp_corner2)
        for i in range(self.corner_num):
            # print(temp_corner1[i], temp_corner2[i])
            err_t = np.mean(np.abs(temp_corner1[i]-temp_corner2[i]))
            
            self.err += err_t

        self.err /= self.corner_num
        return self.err 

    def sort_points_list(self, points_list):
        """
        """
        points_list = sorted(points_list, key=lambda x:x[0]) # x line horizon

        h, v = self.corner_size[0], self.corner_size[1]

        for i in range(v):
            points_list[i*h:(i+1)*h] = sorted(points_list[i*h:(i+1)*h], key=lambda x:x[1])
        
        return points_list




    def sort_by_corners(self, show_flag=False):
        """
        """
        # self.corner =  np.expand_dims(self.corner, 1)
        c_shape = self.corner.shape
        sf_shape = self.sort_ref_corner.shape 
        assert self.corner.shape == self.sort_ref_corner.shape, f'{c_shape} =? {sf_shape}'
        corner_sorted = np.zeros(tuple(self.sort_ref_corner.shape))
        for i in range(self.sort_ref_corner.shape[0]):
            p_t = self.sort_ref_corner[i]
            temp = np.zeros(tuple(self.sort_ref_corner.shape))
            temp[:] = p_t
            err = np.sum((np.power(temp-self.corner,2)),axis=1)
            index_t = np.where(err==err.min())
            corner_sorted[i] = self.corner[index_t[0]]
        
        self.corner = corner_sorted

        img_corner = np.zeros((480,480))
        img_ref = np.zeros((480,480))
        if show_flag:
            err_all = 0
            count = 1
            for i in range(self.corner.shape[0]):
                img_corner[int(self.corner[i][0]), int(self.corner[i][1])] = 255
                img_ref[int(self.sort_ref_corner[i][0]), int(self.sort_ref_corner[i][1])] = 255
                print(self.corner[i],'=?',self.sort_ref_corner[i])
                err_t = np.sum(np.abs(self.corner[i]-self.sort_ref_corner[i]))
                err_all += err_t
                count +=1

            err_all /= count
            print(f'Corner error={err_all}')
            img_corner = Image.fromarray(img_corner)
            img_corner.show()
            img_ref = Image.fromarray(img_ref)
            img_ref.show() 
        return corner_sorted

    def sort_by_gt_data(self, show_flag = False):
        """For Calibration
            Sort self.corner by gt 
        """
        if not self.gt_flag: return None
        self.corner = np.array(self.corner)
        assert self.corner.shape == self.gt.shape

        corner_sorted = np.zeros(tuple(self.gt.shape))
        for i in range(self.gt.shape[0]):
            p_t = self.gt[i]
            temp = np.zeros(tuple(self.gt.shape))
            temp[:] = p_t
            err = np.sum(np.abs(temp-self.corner),axis=1)
            index_t = np.where(err==err.min())
            corner_sorted[i] = self.corner[index_t[0]]
        
        self.corner = corner_sorted
        img_corner = np.zeros((480,480))
        # draw = ImageDraw.Draw(img_corner)
        if show_flag:
            for i in range(self.corner.shape[0]):
                print(self.corner[i],'=?',self.gt[i])
                img_corner[int(self.corner[i][0]), int(self.corner[i][1])] = min(255,i*15)
            img_corner = Image.fromarray(img_corner)
            img_corner.show()
        return corner_sorted

    # need hide
    def run(self, index, distribution_map, corner_num, gt_flag = False, gt_info_name='', corner_size=[6,5], info_flag=False):
        """v 21-04-10 for detection accuracy
        """
        # log_init(DETECTLOGFILE)
        self.load_distribution_map(index, distribution_map, corner_num, corner_size)
        flag, candidates = self.thredshold_locate()
        if not flag:
            cs, s_err = self.sub_pixel_localization(candidates)
            
            if gt_flag:
                gt = self.load_gt_corners_for_analysis(gt_info_name, info_flag=info_flag)
                acc = 0
                count = 0
                for i in range(self.corner_num):
                    err = np.mean(np.abs(cs[i]-gt[i]))
                    if err > 2:
                        err = 0
                    else:
                        count += 1
                    print(err)
                    acc += err
                acc /= count
                # accuracy = np.sum(np.sum(np.sum(np.abs(cs - gt))))
                # accuracy /= self.corner_num
                # accuracy /= 2
                logging.info(f'The sub-pixel accracy={acc}')


    #================================================================= Code Part with outlier rejection

    def thredshold_locate_OR(self, cluster_dist = 12**2, thredshold_height=0.01, step=0.005, max_iter=50):
        """use threshold to locate

        Returns:
            flag: rejection flag - True for rejection
            candidates: putative location

        """
        # start
        locate_number = 0
        candidates = []
        iter_count = 0
        corner_num = self.corner_num
        flag = False
        dist = cluster_dist
        locate_thre = 2;

        while abs(locate_number - corner_num) > locate_thre:
            
            # iter count
            iter_count += 1
            if iter_count > max_iter:
                break

            # height adaptive
            if locate_number < corner_num:
                thredshold_height += step
            else:
                thredshold_height -= (step/4)

            # reset
            locate_number = 0

            # get highest value
            threshold = self.distribution_map.max() - thredshold_height
            index = np.where(self.distribution_map > threshold)

            # choose candidates from index with clustering
            i = 0 # pt for index
            while i < len(index[0]):
                if len(candidates) == 0:
                    candidates.append([index[0][0], index[1][0]])
                    locate_number+=1
                else:
                    # min dist get
                    d = np.Inf
                    for can in candidates:
                        d_t = (index[0][i]-can[0])**2 + (index[1][i]-can[1])**2
                        if d_t < d:
                            d = d_t
                    
                    # clustering
                    if d > dist:
                        candidates.append([index[0][i], index[1][i]])
                        locate_number += 1
                i += 1

        if self.show_flag:
            corner = np.zeros((480,480))
            for can in candidates:
                corner[can[0],can[1]] = 255
            corner = Image.fromarray(corner)
            corner.show()

        l = len(candidates)
        if abs(l-corner_num) < locate_thre:
            # logging.info(f'Well putative located of {self.index} map with {l} candicates')
            pass
        else:
            logging.info(f'Reject Image {self.index} with candidates={l}')
            flag = True
        candidates = self.sort_points_list(candidates)

        return flag, candidates
    

    def sub_pixel_localization_OR(self, candidates):
        """ get & save

        Returns:
            corners: Nx1x2
        """
        corners = []
        Outliers = []
        sigma_err = 0
        outlier_thre = 7
        outlier_count = 0
        flag = True

        for can in candidates:
            cor, s_e = self.__subpixel_localization_for_single_point(can)

            # outlier rejection
            if s_e > outlier_thre:
                Outliers.append(1)
                outlier_count += 1
            else:
                Outliers.append(0)

            sigma_err += s_e
            corners.append(cor)
        
        # corners = self.sort_points_list(corners)
        corners = np.array(corners)
        s = corners.shape
        # if (s - outlier_count) > corner_num:
        #     flag = False # drop this image
        assert(s[0] == len(Outliers))
        sigma_err /= self.corner_num

        # np.save(os.path.join(self.save_path, self.index+'.npy'), corners)

        # logging.info(f'Get sub-pixel corners with shape {s} and sigma error = {sigma_err}')

        self.corner = corners
        # print("corners shape = ", corners.shape)

        return corners, sigma_err, Outliers #, flag

    #++++++++++

    def __draw_corners(self, corners):
        corner = np.zeros((480,480))
        for A in corners:
            corner[int(A[0]), int(A[1])] = 255
        corner = Image.fromarray(corner)
        corner.show()

    def __draw_corners2(self, corners, ref_corners):
        corner = np.zeros((480,480))
        for A in corners:
            corner[int(A[0]), int(A[1])] = 255

        for B in ref_corners:
            corner[int(B[0]), int(B[1])] = 100
        


        corner = Image.fromarray(corner)
        corner.show()



    def opencv_find_corner(self, img):
        """
        """
        self.sort_ref_corner = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (chessboardsize[1],chessboardsize[0]), cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_FAST_CHECK +
                                                cv2.CALIB_CB_FILTER_QUADS)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # cv2.drawChessboardCorners(img, (chessboardsize[1],chessboardsize[0]), corners2, ret)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            for i in range(corners2.shape[0]):
                self.sort_ref_corner.append([corners2[i][0,1],corners2[i][0,0]])
            # self.sort_ref_corner.append([corner[i][0,1],corner[i][0,0]])
        
            self.sort_ref_corner = np.array(self.sort_ref_corner)
            return False
        
        else:
            return True

    def collineation_refinement_OR(self, corners, outliers):
        """
        """
        # print("~\n",corners)
        # print("*\n", self.sort_ref_corner)
        # self.__draw_corners2(corners, self.sort_ref_corner)
        hor_set_size = chessboardsize[1]
        hor_line_sets = []
        vec_set_size = chessboardsize[0]
        vec_line_sets = []

        # hor -> vec in image
        temp_hor_line_set = []
        for i in range(vec_set_size):
            # print("*")
            temp_hor_line_pts = []
            for j in range(hor_set_size):
                ref_cor = self.sort_ref_corner[i*hor_set_size + j]
                # print("**", ref_cor)
                index, flag = self.__get_cloest_corner(ref_cor, corners)
                # print(index)
                if flag and outliers[index[0]] == 0:
                    # print("~", corners[index[0]])
                    temp_hor_line_pts.append(corners[index[0]])
            # print(temp_hor_line_pts)
            line_flag, temp_line = self.__fit_line_ransac(temp_hor_line_pts)
            if not line_flag: return False, None
            # self.__draw_line(temp_line)
            temp_hor_line_set.append(temp_line)

        # vec -> hor in image
        temp_vec_line_set = []
        for i in range(hor_set_size):
            # print("*")
            temp_vec_line_pts = []
            for j in range(vec_set_size):
                ref_cor = self.sort_ref_corner[j*hor_set_size + i]
                # print("**", ref_cor)
                index, flag = self.__get_cloest_corner(ref_cor, corners)
                # print(index)
                if flag and outliers[index[0]] == 0:
                    # print("~", corners[index[0]])
                    temp_vec_line_pts.append(corners[index[0]])
            # print(temp_vec_line_pts)
            line_flag, temp_line = self.__fit_line_ransac(temp_vec_line_pts)
            if not line_flag: return False, None
            # self.__draw_line(temp_line)
            temp_vec_line_set.append(temp_line)
        
        # intersect lines
        final_corners = []
        for hor_line in temp_hor_line_set:
            for vec_line in temp_vec_line_set:
                temp_x, temp_y = self.__intersect_lines(hor_line, vec_line)
                final_corners.append([temp_x, temp_y])

        # for vec_line in temp_vec_line_set:
        #     for hor_line in temp_hor_line_set:
        #         temp_x, temp_y = self.__intersect_lines(hor_line, vec_line)
        #         final_corners.append([temp_x, temp_y])

        # self.__draw_corners2(corners, final_corners)

        # if self.gt_flag:
        #     self.corner = final_corners
        #     final_corners = self.sort_by_gt_data()

        return True, final_corners



    def __draw_line(self, line):
        """
        """
        k = float(line[1]) / float(line[0])
        b = -k * line[2] + line[3]
        img = np.zeros((480,480,3), np.uint8)
        ptStart = (20, int(20*k+b))
        ptEnd = (400, int(400*k+b))
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        cv2.imshow(" ", img)
        cv2.waitKey(0)


    

    def __get_cloest_corner(self, ref_corner, corners):
        """
        """
        temp = np.zeros(tuple(corners.shape))
        temp[:] = ref_corner
        err = np.sum((np.power(temp - corners, 2)), axis=1)
        # print(err)
        if err.min() < 100:
            flag = True
        else:
            flag = False
        return np.where(err==err.min())[0], flag


    def __fit_line_ransac(self, points):
        """
        """
        import math
        iterations = 5000
        k_min = -100
        k_max = 100
        line = [0,0,0,0]
        sigma = 1.0


        points_num = len(points)

        if points_num<2:
            print("less pts")
            return False, line

        bestScore = -1
        for k in range(iterations):
            i1,i2 = random.sample(range(points_num), 2)
            p1 = points[i1]#[0]
            p2 = points[i2]#[0]

            dp = p1 - p2 
            dp *= 1./np.linalg.norm(dp) 

            score = 0
            a = dp[1]/dp[0]
            if a <= k_max and a>=k_min:
                for i in range(points_num):
                    v = points[i][0] - p1
                    dis = v[1]*dp[0] - v[0]*dp[1]
                    # score += math.exp(-0.5*dis*dis/(sigma*sigma))
                    if math.fabs(dis)<sigma:
                        score += 1
            if score > bestScore:
                line = [dp[0],dp[1],p1[0],p1[1]]
                bestScore = score

        return True, line
    
    def __intersect_lines(self, line1, line2):
        """
        """
        assert(line1[0] != 0 and line2[0] != 0)
        k1 = line1[1] / line1[0]
        b1 = -k1 * line1[2] + line1[3]
        k2 = line2[1] / line2[0]
        b2 = -k2 * line2[2] + line2[3]

        x = (b2 - b1) / (k1 - k2 + 1e-5)
        y = k1 * x + b1

        return x, y
        




    def run_OR(self, heatmap, ori_img):
        """
        """
        self.load_distribution_map(0, heatmap, corner_num, chessboardsize)
        flag = self.opencv_find_corner(ori_img)
        if flag:
            print("Drop bad image.")
            return False, None
        

        flag, cans = self.thredshold_locate_OR()
        if flag:
            print("Drop bad image.")
            return False, None
        else:
            corners, se, outliers = self.sub_pixel_localization_OR(cans)
            c_flag, corners2 = self.collineation_refinement_OR(corners, outliers)

        return c_flag, corners2



