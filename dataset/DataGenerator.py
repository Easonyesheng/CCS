"""class used to generate synthesized data for camera calibration"""

import numpy as np
from PIL import Image
import math
from math import sin, cos, pi, floor, ceil, tan, sqrt
import random
import os
import glob
from tqdm import tqdm
import copy
from PIL import ImageFilter
import cv2

from utils.utils import *
from settings.settings import *

class Checkboard(object):
    """Workflow
    functions:
        1. world_location_load: get the ori points loaction in world coordinate
        2. camera_load: load camera parameters
        3. shoot: from world coordinate to camera coordinate
        4. project: from camera coordinate to image coordinate
    NOTATION:
        Corners are [x,y,1]^T -> img[x,y]
    """
    def __init__(self, save_path, name, corner_size=[5,5], img_size = 480):
        """
        """
        
        self.name = name
        # log_init(os.path.join(LOGFILEPATH, 'checkboard'+name+'.txt'))
        self.width = corner_size[0]
        self.height = corner_size[1]
        self.checkboard_size = corner_size
        self.img_size = img_size

        self.img_save_path = save_path[0]
        self.info_save_path = save_path[1]


        if len(save_path) > 2:
            self.heatmap_save_path = save_path[2]

        if len(save_path)>3:
            self.dist_save_path = save_path[3]
            self.ori_corner_save_path = save_path[4]
            self.dist_corner_save_path = save_path[5]

        self.camera_load_flag = False

    def world_location_load(self, distance=0.0):
        """load checkboard world coordinate points location
        
        """
        self.z = 0.0
        X = range(self.width)
        Y = range(self.height)
        self.world_points = []
        for i in X:
            for j in Y:
                self.world_points.append([float(i),float(j),self.z, 1.0])

        return self.world_points
    
    def camera_load(self, fx=150, fy=150, px=100, py=100):
        """
        """
        self.K = np.matrix([[fx,0,px],
                            [0,fy,py],
                            [0,0, 1]])

        self.camera_load_flag = True
        
        return self.K

    def __move_checkboard(self, d=5, phi=0, theta=0):
        """move the checkboard accroding to d, theta(up the x-z plane), phi(in the x-z plane)
            The move converts to T=[tx,ty,tz] as 
                tx = d * cos(phi) * sin(theta)
                ty = d * sin(phi)
                tz = d * cos(phi) * cos(theta)
        Args:
            phi, theta as degree
        """

        theta = theta*pi/180 # up the x-z plane
        phi = phi*pi/180 #in x-z plane

        tx = d*cos(phi)*sin(theta)
        ty = d*sin(phi)
        tz = d*cos(phi)*cos(theta)

        T = [tx, ty, tz]
        T = np.matrix(T).T

        return T

    def move_checkboard(self, d=5, phi=0, theta=0):
        """move the checkboard accroding to d, theta(in the y-z plane), phi(in the x-z plane)
            # The move converts to T=[tx,ty,tz] as 
            #     tx = d * cos(phi) * sin(theta)
            #     ty = d * sin(phi)
            #     tz = d * cos(phi) * cos(theta)
            Modified: d is z
                tx = d * tan(phi)
                ty = d * sqrt(tan(theta)**2 - tan(phi)**2)
                tz = d
        Args:
            phi, theta as degree
        """

        theta = theta*pi/180 # up the x-z plane
        phi = phi*pi/180 #in x-z plane

        # tx = d*cos(phi)*sin(theta)
        # ty = d*sin(phi)
        # tz = d*cos(phi)*cos(theta)

        tx = d*tan(phi)
        ty = d*tan(theta)
        tz = d

        T = [tx, ty, tz]
        self.T = np.matrix(T).T

        return self.T
    
    def __rotate_checkboard(self, x=0, y=0, z=0):
        """rotate the checkboard 
        Args:
            x is the degree rotate by x axis
            ...
        """
        x = x*pi/180
        y = y*pi/180
        z = z*pi/180


        #x
        m_x = [[1,0,0],
            [0,cos(x),sin(x)],
            [0, -sin(x), cos(x)]]

        #y
        m_y = [[cos(y), 0, sin(y)],
            [0,1,0],
            [-sin(y), 0, cos(y)]]

        #z
        m_z = [[cos(z),sin(z),0],
            [-sin(z), cos(z),0],
            [0,0,1]]

        m_x = np.matrix(m_x)
        m_y = np.matrix(m_y)
        m_z = np.matrix(m_z)

        #ZYX
        R = m_x*(m_y*m_z)

        return R

    def rotate_checkboard(self, x=0, y=0, z=0):
        """rotate the checkboard 
        Args:
            x is the degree rotate by x axis
            ...
        """
        x = x*pi/180
        y = y*pi/180
        z = z*pi/180


        #x
        m_x = [[1,0,0],
            [0,cos(x),sin(x)],
            [0, -sin(x), cos(x)]]

        #y
        m_y = [[cos(y), 0, sin(y)],
            [0,1,0],
            [-sin(y), 0, cos(y)]]

        #z
        m_z = [[cos(z),sin(z),0],
            [-sin(z), cos(z),0],
            [0,0,1]]

        m_x = np.matrix(m_x)
        m_y = np.matrix(m_y)
        m_z = np.matrix(m_z)

        #ZYX
        self.R = m_x*(m_y*m_z)

        return self.R
    
    def get_homography(self):
        """
        H = K[r1 r2 t]
        """
        r1 = self.R[:,0]
        r2 = self.R[:,1]
        t = self.T
        H = np.hstack((r1,r2))
        H = np.hstack((H,t))
        self.H = self.K*H

        return self.H

    def get_fundamental(self):
        """Fundamental Matrix
        F = K^-T [t]_x R K^-1 
        """
        K_I = self.K.I
        K_IT = K_I.T
        
        t_x = np.matrix([
            [0,         0, self.T[1,0]],
            [0,      0,    -self.T[0,0]],
            [-self.T[1,0], self.T[0,0],   0      ],
        ])

        R = self.R
        self.F = K_IT * t_x * R * K_I

        return self.F

    def shoot(self):
        """
        """
        inside_img_flag = True
        P = np.hstack((self.R, self.T))
        self.P = self.K * P
        self.img_points = []
        for p_wc in self.world_points:
            p_wc = np.matrix(p_wc)
            p_ic = self.P * p_wc.T
            p_ic[0] /= p_ic[2]
            p_ic[1] /= p_ic[2]
            p_ic[2] /= p_ic[2]
            if p_ic[0] < 5 or p_ic[0] > self.img_size-5 or p_ic[1] < 5 or p_ic[1] > self.img_size-5:
                inside_img_flag = False
                break
            self.img_points.append(p_ic)
        
        return inside_img_flag, self.P, self.img_points
    
    def shoot_without_homo(self):
        """
        """
        inside_img_flag = True
        P = np.hstack((self.R, self.T))
        self.P = self.K * P
        self.img_points = []
        for p_wc in self.world_points:
            p_wc = np.matrix(p_wc)
            p_ic = self.P * p_wc.T
        
            if p_ic[0]/p_ic[2] < 5 or p_ic[0]/p_ic[2]  > self.img_size-5 or p_ic[1]/p_ic[2]  < 5 or p_ic[1]/p_ic[2]  > self.img_size-5:
                inside_img_flag = False
                break
            self.img_points.append(p_ic)
        
        return inside_img_flag, self.P, self.img_points

    def print_img(self, show_flag=False):
        """
        """
        H = self.img_size
        W = self.img_size
        sigma = 0.01
        heatmap = np.zeros([W,H])

        # find the right boundary
        x_min = 480
        y_min = 480
        x_max = 0
        y_max = 0

        # self.add_asymmetric_label()

        for corner in self.img_points:
            yc, xc, _ = int(corner[0]), int(corner[1]), float(corner[2])
            if yc > y_max:
                y_max = yc
            if yc < y_min:
                y_min = yc
            
            if xc > x_max:
                x_max = xc
            if xc < x_min:
                x_min = xc
        if x_min < 16:
            x_min = 16
        if y_min < 16:
            y_min = 16
        
        if x_max > 464:
            x_max = 464
        if y_max > 464:
            y_max = 464

        print('Drawing Heatmap...')
        for x in range(x_min-15, x_max+15):
            for y in range(y_min-15, y_max+15):
                r = np.Infinity
                for corner in self.img_points:
                    yc, xc, _ = float(corner[0]), float(corner[1]), float(corner[2])

                    x_norm = x / W
                    y_norm = y / H

                    x_center_cor = xc / W
                    y_center_cor = yc / H

                    r_temp = (x_norm - x_center_cor)*(x_norm - x_center_cor) + (y_norm - y_center_cor)*(y_norm - y_center_cor)
                    r = min(r_temp, r)
                
                if r > 20:
                    continue
                # print(r)
                Gaussion = 1/math.sqrt(2*np.pi*sigma*sigma) * np.exp(-(r)/(2*sigma*sigma))
                # print(Gaussion)
                heatmap[x,y] += Gaussion

        # heatmap /= (np.max(heatmap)+1e-5)
        heatmap *= math.sqrt(2*np.pi*sigma*sigma)
        heatmap *= 255
        img = Image.fromarray(heatmap)

        # img = self.draw_asymmetric(img, self.checkboard_size, self.img_points)

        if show_flag:
            img.show()
        
        self.img = img.convert('RGB')
        return self.img
        
    def draw_asymmetric(self, img, corner_size, points):
        """
        """
        pass

    def add_asymmetric_label(self):
        """
        """
        p1 = self.img_points[0]
        p2 = self.img_points[1]
        dxy = p2 - p1
        p_label = p1 - dxy
        self.img_points.append(p_label)
    
    def run(self, camera_parameters, move_parameters, rotate_parameters, save_flag=False, show_flag = False):
        """script
        """
        

        print('checkboard '+self.name+' is generating...')

        world_points = self.world_location_load()
        
        fx = camera_parameters['fx'] 
        fy = camera_parameters['fy']
        px = camera_parameters['px'] 
        py = camera_parameters['py'] 

        K = self.camera_load(fx, fy, px, py)

        d = move_parameters['d']
        phi = move_parameters['phi']
        theta = move_parameters['theta']

        t = self.move_checkboard(d, phi, theta)

        x = rotate_parameters['x']
        y = rotate_parameters['y']
        z = rotate_parameters['z']

        R = self.rotate_checkboard(x,y,z)
        
        H = self.get_homography()
        F = self.get_fundamental()

        inside_img_flag, P, img_points = self.shoot()

        if inside_img_flag:
            img = self.print_img(show_flag)
            print(f"""
                    parameters are: \n
                    {camera_parameters} \n
                    {move_parameters} \n
                    {rotate_parameters} \n
                """)

            if save_flag:
                info_save = {
                    'checkboard_size': self.checkboard_size,
                    'K': K,
                    't': t,
                    'R': R,
                    'P': P,
                    'H': H,
                    'F': F,
                    'img_points': img_points,
                    'world_points': world_points, # list[[X,Y,0,1]]
                    'img_save_path' : self.img_save_path, #list[matrix[x,y,1]]
                    'info_save_path': self.info_save_path
                }

                np.save(os.path.join(self.info_save_path, self.name+'.npy'), info_save)
                img.save(os.path.join(self.img_save_path, self.name+'.jpg'))
        else:
            print(f'Out image!')
        
        return inside_img_flag

    def deside_the_border(self):
        """
        """
        points_fix = self.img_points
        checkboard_block_size =ceil(-1*(points_fix[0][0,0]/points_fix[0][2,0] - points_fix[1][1,0]/points_fix[1][2,0]))

        start_w = ceil(points_fix[0][0,0]/points_fix[0][2,0]) - checkboard_block_size
        end_w = ceil(points_fix[-1][0,0]/points_fix[-1][2,0]) + checkboard_block_size
        start_h = ceil(points_fix[0][1,0]/points_fix[0][2,0]) - checkboard_block_size
        end_h = ceil(points_fix[-1][1,0]/points_fix[-1][2,0]) + checkboard_block_size

        return start_h, start_w, end_h, end_w, checkboard_block_size

    def draw_the_whiteside(self, img, start_h, start_w, end_h, end_w, x_side_l = 10, y_side_l = 20):
        """
        """
        img[start_w-x_side_l:start_w,start_h-y_side_l:end_h+y_side_l]=255
        img[end_w:end_w+x_side_l,start_h-y_side_l:end_h+y_side_l]=255
        img[start_w-x_side_l:end_w+x_side_l, start_h-y_side_l:start_h] = 255
        img[start_w-x_side_l:end_w+x_side_l, end_h:end_h+y_side_l] = 255

        return img
    
    def draw_the_blackdot(self, img, start_h, start_w, checkboard_block_size):
        """
        """
        from PIL import ImageDraw
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        draw.ellipse((start_w,start_h,start_w+checkboard_block_size-5, start_h+checkboard_block_size-5), fill=(0,0,0))
        img = img.convert('L')
        return img

    def draw_fix_checkboard(self, d=10, show_flag = False, fusion_flag=False):
        """Draw fix checkboard picture
        Conditions:
            R = E
            t = [0,0,d].T : D IS FIXED = 10
            K: self.K
        Workflow:
            1. shoot to get img_points
            2. determine the start and end value in X/Y axis, and the block size
            3. draw white side and white/black block
        """
        assert d == 10
        self.world_location_load()
        if not self.camera_load_flag: 
            print("Need Load Camera Parameters!")
            return 
        # step 1
        self.rotate_checkboard()
        tf = self.move_checkboard(d=d)
        inside_flag, _ , _  = self.shoot_without_homo()

        if not inside_flag:
            print("Out Image")
            return

        # step 2
        img_fix = np.ones((self.img_size, self.img_size)) # ones for fusion
        start_h, start_w, end_h, end_w, checkboard_block_size = self.deside_the_border()
        img_fix = self.draw_the_whiteside(img_fix, start_h, start_w, end_h, end_w)
        # start drawing
        w = start_w
        h = start_h
        flag = True

        while(w < end_w):
            while(h < end_h):
                for i in range(w, w + checkboard_block_size):
                    for j in range(h, h + checkboard_block_size):
                        if flag:
                            img_fix[i,j] = 255
                        else:
                            img_fix[i,j] = 0

                h += checkboard_block_size
                flag = not flag

            # print(flag%2, start_w, start_h)

            h = start_h
            w += checkboard_block_size
            if self.height%2 != 0:
                flag = not flag
        
        if fusion_flag:
            import random
            bg_img_list = glob.glob(os.path.join(FUSIONPATH,'*.jpg'))
            img_index = random.randint(0,len(bg_img_list)-1)
            bg_img = cv2.imread(bg_img_list[img_index], 0)
            bg_img = bg_img.astype(np.uint8)
            bg_img = cv2.resize(bg_img, (480,480))
            index_255 = np.where(img_fix > 1)
            img_homo_after_fusion = img_fix*bg_img
            img_homo_after_fusion[index_255] = img_fix[index_255]
            img_fix = img_homo_after_fusion


        img_fix = Image.fromarray(img_fix)

        # img_fix = self.draw_the_blackdot(img_fix, start_h, start_w, checkboard_block_size)

        if show_flag:
            img_fix.show()
        
        return img_fix, tf

    def draw_move_checkboard(self, img_fix, img_bg, config, save_flag=False, show_flag=False, fusion_flag =False, heatmap_flag = False, dist_flag=False, dist_k=[1,0.1,0], uneven_light_flag = False):
        """
        Args:
            config['x/y/z']:Rm
            config['d/phi/theta']:tm 3 x 1 matirx
            config['tf']:tf 3 x 1 matirx
            img_bg: background - pil_img_'L'
        """
        if not self.camera_load_flag:
            print("No Camera Load")
            return

        print('checkboard '+self.name+' with img is generating...')

        world_points = self.world_location_load()
        
        x = config['x']
        y = config['y']
        z = config['z']

        Rm = self.rotate_checkboard(x,y,z)

        d = config['d']
        phi = config['phi']
        theta = config['theta']

        tm = self.move_checkboard(d, phi, theta)

        tf = config['tf']
        df = tf[2,0]

        K = self.K
        K_I = self.K.I

        KRK_ = K*Rm*K_I
        KRtf = K*Rm*tf
        Kt = K*tm

        inside_img_flag, P, img_points = self.shoot()

        if inside_img_flag:
            img_move = self.draw_checkboard_by_interp(img_fix, KRK_, KRtf, Kt, df, show_flag)

            if dist_flag:
                img_dist, ori_corner, dist_corner = self.apply_distortion(np.array(img_move),k=dist_k)

            if uneven_light_flag:
                img_move = self.apply_uneven_light(img_move)
                img_move = Image.fromarray(img_move)
                img_move = img_move.convert('L')

            print(f"""
                    parameters are: \n
                    {K} \n
                    {Rm} \n
                    {tm} \n
                """)

            if heatmap_flag:
                assert len(self.heatmap_save_path) > 1 
                heatmap = self.draw_heatmap(img_points)


            if save_flag:
                info_save = {
                    'checkboard_size': self.checkboard_size,
                    'K': K,
                    't': tm,
                    'R': Rm,
                    'img_points': img_points,
                    'world_points': world_points,
                    'img_save_path' : self.img_save_path,
                    'info_save_path': self.info_save_path
                }

                np.save(os.path.join(self.info_save_path, self.name+'.npy'), info_save)
                #fusion
                # if fusion_flag:
                #     img_move = self.fusion_with_bg(img_move, img_bg)
                #     img_move = Image.fromarray(img_move).convert('RGB')
                #     if show_flag:
                #         img_move.show()
                img_move.save(os.path.join(self.img_save_path, self.name+'.jpg'))
                if dist_flag:
                    img_dist.save(os.path.join(self.dist_save_path, self.name+'.jpg'))
                    np.save(os.path.join(self.ori_corner_save_path, self.name+'.npy'), ori_corner)
                    np.save(os.path.join(self.dist_corner_save_path, self.name+'.npy'), dist_corner)
                    


                np.save(os.path.join(self.info_save_path, self.name+'.npy'), info_save)
                np.save(os.path.join(self.heatmap_save_path, self.name+'.npy'), heatmap)

        else:
            print(f'Out image!')
        
        return inside_img_flag
    
    def fusion_with_bg(self, img, img_bg):
        """
        Args:
            all for pil
        """
        img = np.array(img.convert('L'), dtype=np.uint8)
        img_bg = img_bg.convert('L')
        img_bg = img_bg.resize((self.img_size, self.img_size))
        img_bg = np.array(img_bg, dtype=np.uint8)
        index_255 = np.where(img != 1)
        img_after_fusion = img*img_bg
        img_after_fusion[index_255] = img[index_255]

        return img_after_fusion

    def draw_checkboard_by_interp(self, img_fix, KRK_, KRtf, Kt, df, show_flag=False):
        """
        """
        img_move = np.zeros((self.img_size, self.img_size)) #  use ones for fusion
        img_f_array = np.array(img_fix.convert('L'))

        for x_f in range(self.img_size-2):
            for y_f in range(self.img_size-2):
                pf_h = np.matrix(np.zeros((3,1)))
                X_f = x_f*df
                Y_f = y_f*df
                pf_h[0,0] = X_f
                pf_h[1,0] = Y_f
                pf_h[2,0] = df
                pm_h = KRK_*pf_h - KRtf + Kt
                x_m = pm_h[0,0] / pm_h[2,0]
                y_m = pm_h[1,0] / pm_h[2,0]
                
                if x_m > self.img_size-1 or y_m > self.img_size-1 or x_m < 0 or y_m < 0:
                    continue
                
                x_m_int = math.floor(x_m)
                y_m_int = math.floor(y_m)
                
                dx = x_m - x_m_int
                dy = y_m - y_m_int
                
                img_move[x_m_int,y_m_int] = (1-dx)*(1-dy)*img_f_array[x_f,y_f] + dy*(1-dx)*img_f_array[x_f,y_f+1] + \
                                               (1-dy)*dx*img_f_array[x_f+1,y_f] + dy*dx*img_f_array[x_f+1,y_f+1]
                img_move[x_m_int,y_m_int+1] = (1-dx)*(1-dy)*img_f_array[x_f,y_f+1] + dy*(1-dx)*img_f_array[x_f,y_f+2] + \
                                        (1-dy)*dx*img_f_array[x_f+1,y_f+1] + dx*dy*img_f_array[x_f+1,y_f+2]
                img_move[x_m_int+1,y_m_int] = (1-dx)*(1-dy)*img_f_array[x_f+1,y_f] + dy*(1-dx)*img_f_array[x_f+1,y_f+1] + \
                                        (1-dy)*dx*img_f_array[x_f+2,y_f] + dx*dy*img_f_array[x_f+2,y_f+1]
                img_move[x_m_int+1,y_m_int+1] = (1-dx)*(1-dy)*img_f_array[x_f+1,y_f+1] + dy*(1-dx)*img_f_array[x_f+1,y_f+2] + \
                                        (1-dy)*dx*img_f_array[x_f+2,y_f+1] + dx*dy*img_f_array[x_f+2,y_f+2]

        img_m = Image.fromarray(img_move)
        img_m = img_m.convert('RGB')
        img_m = img_m.filter(ImageFilter.GaussianBlur(radius=1.6)) # 1.6 with distance=6~8 is good
        img_m= img_m.convert('L')
        if show_flag:
            img_m.show()

        return img_m

    def draw_heatmap(self, img_points):
        """
        """
        heatmap = np.zeros((self.img_size, self.img_size))
        corners = img_points
        W = self.img_size
        H = self.img_size
        sigma = 0.01

        for x in range(W):
            for y in range(H):
                r = np.Infinity
                for corner in corners:
                    # print(corner)
                    # corners = [[240,240,1]]
                   
                    xc, yc, _ = float(corner[0]), float(corner[1]), float(corner[2])
                    
                    x_norm = x / W
                    y_norm = y / H

                    x_center_cor = xc / W
                    y_center_cor = yc / H

                    r_temp = (x_norm - x_center_cor)*(x_norm - x_center_cor) + (y_norm - y_center_cor)*(y_norm - y_center_cor)
                    r = min(r_temp, r)
                
                if r > 50:
                    continue
                # print(r)
                # Gaussion = 1/math.sqrt(2*np.pi*sigma*sigma) * np.exp(-(r)/(2*sigma*sigma))
                Gaussion = np.exp(-(r)/(2*sigma*sigma))
                # print(Gaussion)
                heatmap[x,y] += Gaussion

        # heatmap /= (np.max(heatmap)+1e-5)
        # heatmap *= math.sqrt(2*np.pi*sigma*sigma)
        # heatmap *= 255
        # heatmap[int(corners[0][0]), int(corners[0][1])] = 255
        # img = Image.fromarray(heatmap)
        # img.show()

        return heatmap

    def apply_distortion(self, img, W = 480, k=[1, 0.5, 0]):
        """apply radial distortion to image
            coff = k0+  k1 x r^2 + k2 x r^4
        Args:
            img: current image
            k: distortion parameter
        Returns:
            disted image
            ori corners
            dist corner
        """
        img = np.array(img)
        img_mat = np.zeros((480,480), dtype=np.uint8)
        # print(img.dtype)
        dis_center_x = W/2
        dis_center_y = W/2
        H = W

        for x in range(W-2):
            for y in range(H-2):
                x_nom = (x - dis_center_x)/ (W)
                y_nom = (y - dis_center_y)/ (W)
                r = (x_nom)*(x_nom) + (y_nom)*(y_nom)
                # r = r**(0.5)

                coff = (k[0]+k[1]*(r) + k[2]*(r**2))
                x_d = ((coff)*((x-W/2)) + W/2)
                y_d = ((coff)*((y-H/2)) + H/2)
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


        img_show = Image.fromarray(img_mat)
        img_show = img_show.convert('L')

        # corner 
        ori_corner = []
        dist_corner = []
        for i in range(0,W-2,5):
            for j in range(0,W-2,5):
                ori_corner.append([i,j,1])
                i_nom = (i - dis_center_x) / W
                j_nom = (j - dis_center_y) / W

                r = i_nom**2 + j_nom**2

                # coff = 1 + k*r
                coff = (k[0]+k[1]*(r) + k[2]*(r**2))
                i_d = coff*(i - W/2) + W/2
                j_d = coff*(j - W/2) + W/2

                # ori_corner.append([i, j, 1])
                dist_corner.append([i_d,j_d,1])


        return img_show, ori_corner, dist_corner
  
    def apply_uneven_light(self, img):
        """Apply uneven illumination on image

        """
        img = np.array(img)
        img_light = self.__draw_uneven_light(img.shape)

        img_uneven_light = img + img_light
        img = (img_uneven_light - np.min(img_uneven_light)) / (np.max(img_uneven_light)-np.min(img_uneven_light))
        img = img*255

        return img

    def __draw_uneven_light(self, img_size):
        """
        """
        [W, H] = img_size
        # sigma = 0.2 + np.random.randn()/10 * 3
        # sigma = 0.2 if sigma<0.2 else sigma
        sigma = 0.4
        light = 255
        center_w = np.random.randint(W//8, W*7//8)
        center_h = np.random.randint(H//4, H*3//4)

        img_light = np.zeros(tuple(img_size))

        for x in range(W):
            for y in range(H):
                x_norm = x / W
                y_norm = y / H

                x_center_cor = center_w / W
                y_center_cor = center_h / H

                r = (x_norm - x_center_cor)*(x_norm - x_center_cor) + (y_norm - y_center_cor)*(y_norm - y_center_cor)
                Gaussion = 1/math.sqrt(2*np.pi*sigma*sigma) * np.exp(-(r)/(2*sigma*sigma))*light
                img_light[x,y] = Gaussion
        
        return img_light