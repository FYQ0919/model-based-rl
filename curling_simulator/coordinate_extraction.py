import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from . import curling_sim
from enum import Enum
from .config import *  # 重要的参数
from .utils import *   # 辅助函数与类

low_s = 80
def red_hsv(bgr_img):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    low1 = np.array([0,low_s,46])
    high1 = np.array([10,255,255])

    low2 = np.array([156,low_s,46])
    high2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv_img, low1, high1)
    mask2 = cv2.inRange(hsv_img, low2, high2)

    mask = mask1 + mask2
    detected = cv2.bitwise_and(bgr_img, bgr_img, mask = mask)
    # shape_list = detected.shape
    # detected[int(shape_list[0]*4/5):, :, :] = 0
    return detected

def yellow_hsv(bgr_img):

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    low = np.array([10,low_s,46])
    high = np.array([40,255,255])

    mask = cv2.inRange(hsv_img, low, high)

    detected = cv2.bitwise_and(bgr_img, bgr_img, mask = mask)
    # shape_list = detected.shape
    # detected[int(shape_list[0]*4/5):, :, :] = 0
    return detected

def get_center(bgr_img):
    kernel = np.ones(shape=[4,4],dtype=np.uint8)
    bgr_img = cv2.erode(bgr_img, kernel=kernel, iterations=1)
    bgr_img = cv2.dilate(bgr_img, kernel, iterations=1)

    # bgr_img = cv2.resize(bgr_img,(320,640))
    # kernel = np.ones(shape=[6,6],dtype=np.uint8)
    # bgr_img = cv2.erode(bgr_img, kernel=kernel, iterations=1)
    # binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    gray_image = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
    _,binary_image = cv2.threshold(gray_image,7,255,cv2.THRESH_BINARY)
    # print(binary_image.shape)
    # binary_image = cv2.resize(binary_image, (64, 128))

    
    # binary_image = cv2.resize(binary_image, (320, 640))
    _,binary_image = cv2.threshold(binary_image,10,0,cv2.THRESH_TOZERO)
    # cv2.imshow('1', binary_image)
    # cv2.waitKey()
    # out = cv2.resize(bgr_img, (320, 640)).copy()
    # center_map = np.zeros((128, 64, 3))
    # out = np.ones((320,640,3))*0
    stones_pos = []

    _,contours,hierarchy = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for idx,obj in enumerate(contours):
        area = cv2.contourArea(obj)
        # print(area)
        if area>100:
            # print(area)
            # print(obj.shape)
            # cv2.drawContours(out, obj, -1, (255, 0, 0), 1)
            (x, y), radius = cv2.minEnclosingCircle(obj)
            # print(radius)
            center = (int(x), int(y))
            radius = int(radius)
            # cv2.circle(out, center, radius, (0, 255, 255), 1)
            
            h0,w0 = bgr_img.shape[:2]
            pos = [valid_pos_min[0] + (1-y/h0)/0.7229*l3, (1-x/w0)*width]
            stones_pos.append(pos)

            # center_map[int(41*128/194), int(52*64/104)] = [0, 255, 0]
            # if color == 'r':
            #     center_map[int(y*128/640), int(x*64/320)] = [0, 0, 255]
            # if color == 'y':
            #     center_map[int(y*128/640), int(x*64/320)] = [0, 255, 255]   

    # cv2.imshow('1', out)
    # cv2.waitKey()
    # return out
    # cv2.imshow('1', center_map)
    # cv2.waitKey()
    stones_pos = np.array(stones_pos, dtype=np.float32)
    return stones_pos
    
def get_coor_map(folder_path):
    folder = os.listdir(folder_path)
    for file in folder:
        if(file.split('.')[-1] == "png"):
            img = cv2.imread(folder_path + file)
            red_detected = red_hsv(img)
            yellow_deteced = yellow_hsv(img)

            red_center = get_center(red_detected, 'r')
            yellow_center = get_center(yellow_deteced, 'y')
            center = red_center + yellow_center

            save_folder = folder_path + "/coordinates/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            save_path = save_folder + file
            # print(save_path)
            cv2.imwrite(save_path, center)

def get_position(bgr_img):
    red_detected = red_hsv(bgr_img)
    yellow_deteced = yellow_hsv(bgr_img)

    stones_pos_r = get_center(red_detected)
    stones_pos_y = get_center(yellow_deteced)
    # center = red_center + yellow_center
    return stones_pos_r, stones_pos_y

