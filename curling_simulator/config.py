# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import cv2
import torch
import taichi as ti

from enum import Enum

ti_real = ti.f32
ti_int = ti.i32

# Parameter for Simulation
dt = 0.01 # 0.001
print(f"[Simulation] 当前 dt={dt}s, 请注意精度和速度的平衡!")

# dt = 0.01 # 0.001
# print(f"[IMPORTANT] 当前 dt={dt}s, 仅供debug使用")

max_steps = 20000

max_time_length = dt * max_steps
print(f"[Simulation] 单次shot最长仿真时间={max_time_length:.2f}s")
assert max_time_length > 30

vis_interval = 10 # 10
# rendering = True

# Definition for ice sheet (in meter)
l1 = 1.2192 #hack to backboard
l2 = 3.6576 #tee to hack
l3 = 6.4008 #hog line to tee
l4 = 21.9456 #middle

r1 = 0.1524 #inner blue
r2 = 0.6096 #outter blue
r3 = 1.2192 #radius of inner red
r4 = 1.8288 #radius of outter red

hack_length = 0.4572

# 英尺(m)
foot = 0.3048

# 场地尺寸
length = (l1 + l2 + l3) * 2 + l4
width = 4.3180
# 有效球的区域的坐标范围，用于observation的normalization
valid_pos_min = np.array([l1 + l2 + l3 + l4, 0.0])
valid_pos_max = np.array([l1 + l2 + l3 + l4 + l3 + r4, width])
valid_pos_range = valid_pos_max - valid_pos_min

# GUI
scale = 30 # 40
gui_length = int(length * scale) + 1
gui_width = int(width * scale) + 1


# 大本营位置
center_pos = [l1+l2+l3+l4+l3, width/2]

# 每局冰壶数量，包含2个固定壶
# n_curling_stones = 12 
n_curling_stones = 16
n_stones_player = n_curling_stones//2
# 每场比赛的局数 number of ends
n_ends = 8

# 冰壶参数  
radius = 0.145531
elasticity = 0.8 # Need to be adjusted
miu = 1/125
r = 0.0625
R = 0.14
r_p = 0.117
g = 9.81

static_velocity = 2e-2

# pivot时间占比
f_ = 0.00037

# 记录冰壶有效性状态 
STATE_READY = 0     # 尚未击打(默认)  
STATE_VALID = 1     # 在场上,有效      
STATE_INVALID = 2   # 无效球，已被击打但不在场上
invalid_fill_value = -1.0


# 网络输入
network_input_size = (4, 128, 128)
# simplified_input_size = (4, 128, 128)



