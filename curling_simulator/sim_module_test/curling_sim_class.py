# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import torch

import taichi as ti
from enum import Enum

from config import *

# taichi 常用数据类型
ti_real = real = ti.f32
ti_int = ti.i32

class CurlingSimuEngine(object):
    """冰壶底层仿真实现"""
    def __init__(self):
        # Inilization for Taichi
        ti.init(arch=ti.cpu, default_fp=real, flatten_if=True) # ti.cuda 

        # 冰壶shot仿真相关共享变量
        # scalar = lambda: ti.field(dtype=real)
        vec = lambda: ti.Vector.field(2, dtype=real)
        self.x = vec()  # position
        self.v = vec()  # velocity
        self.x_inc = vec()
        self.impulse = vec()
        self.vas = ti.field(shape=(max_steps, n_curling_stones), dtype=ti_int) # 有效性状态
        ti.root.dense(ti.i, max_steps).dense(ti.j, n_curling_stones).place(self.x, self.v, self.x_inc, self.impulse) 
    
    @ti.kernel
    def func(self):
        for t, i in ti.ndrange(max_steps, n_curling_stones):
            self.impulse[t, i] = ti.Vector([1.0, 1.0])
            self.x_inc[t, i] = ti.Vector([1.0, 1.0])


# 打开GUI窗口
if rendering:
    scale = 30 # 40
    gui_length = int(length * scale) + 1
    gui_width = int(width * scale) + 1
    gui = ti.GUI("Curling", (gui_length, gui_width), background_color=0xEEEEEE)

# @ti.kernel
# def clear():
#     for t, i in ti.ndrange(steps, n_curling_stones):
#         impulse[t, i] = ti.Vector([0.0, 0.0])
#         x_inc[t, i] = ti.Vector([0.0, 0.0])

@ti.func
def collide_pair(t, i, j):
    imp = ti.Vector([0.0, 0.0])
    x_inc_contrib = ti.Vector([0.0, 0.0])
    if i != j:
        dist = (x[t, i] + dt * v[t, i]) - (x[t, j] + dt * v[t, j])
        dist_norm = dist.norm()
        rela_v = v[t, i] - v[t, j]
        if dist_norm < 2 * radius:
            dir = ti.Vector.normalized(dist)
            projected_v = dir.dot(rela_v)

            if projected_v < 0:
                imp = -(1 + elasticity) * 0.5 * projected_v * dir
                toi = (dist_norm - 2 * radius) / min(
                    -1e-3, projected_v)  # Time of impact
                x_inc_contrib = min(toi - dt, 0) * imp
    x_inc[t + 1, i] += x_inc_contrib
    impulse[t + 1, i] += imp

@ti.kernel
def collide(t: ti.i32):
    for i in range(n_curling_stones):
        for j in range(i):
            collide_pair(t, i, j)
    for i in range(n_curling_stones):
        for j in range(i + 1, n_curling_stones):
            collide_pair(t, i, j)

@ti.kernel
def time_integrate(t: ti.i32):
    for i in range(n_curling_stones):
        v[t, i] = v[t - 1, i] + impulse[t, i]
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]

        a = 1.0
        if v[t, i][0] > a * dt:
            v[t, i][0] -= a * dt
        elif v[t, i][0] < - a * dt:
            v[t, i][0] += a * dt
        else:
            v[t, i][0] = 0.0

        w = 0.01
        if v[t, i][1] > w * dt:
            v[t, i][1] -= w * dt
        elif v[t, i][1] < - w * dt:
            v[t, i][1] += w * dt
        else:
            v[t, i][1] = 0.0

def forward(visualize=False):

    interval = vis_interval

    for t in range(1, max_steps):
        
        collide(t - 1)
        time_integrate(t)

        if (t + 1) % interval == 0 and visualize:
            pixel_radius = int(radius*scale)+1
            
            # Tee
            gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r4*scale), color=0x000080)
            gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r3*scale), color=0xEEEEEE)
            gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r2*scale), color=0xDC143C)
            gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r1*scale), color=0xEEEEEE)

            gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r4*scale), color=0x000080)
            gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r3*scale), color=0xEEEEEE)
            gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r2*scale), color=0xDC143C)
            gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r1*scale), color=0xEEEEEE)

            # Tee line
            gui.line(((l1+l2)*scale/gui_length, 0), ((l1+l2)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)
            gui.line(((l1+l2+l3+l4+l3)*scale/gui_length, 0), ((l1+l2+l3+l4+l3)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)

            # Centre line
            gui.line((l1*scale/gui_length, width/2*scale/gui_width), ((length-l1)*scale/gui_length, width/2*scale/gui_width), radius=1, color=0xCD5C5C)

            # Hog line
            gui.line(((l1+l2+l3)*scale/gui_length, 0), ((l1+l2+l3)*scale/gui_length, width*scale/gui_width), radius=4, color=0xDC143C)
            gui.line(((l1+l2+l3+l4)*scale/gui_length, 0), ((l1+l2+l3+l4)*scale/gui_length, width*scale/gui_width), radius=4, color=0xDC143C)

            # Hack
            gui.line((l1*scale/gui_length, 0.5-hack_length/2*scale/gui_width), (l1*scale/gui_length, 0.5+hack_length/2*scale/gui_width), radius=5, color=0x000000)
            gui.line(((length-l1)*scale/gui_length, 0.5-hack_length/2*scale/gui_width), ((length-l1)*scale/gui_length, 0.5+hack_length/2*scale/gui_width), radius=5, color=0x000000)

            for i in range(n_curling_stones):
                if i % 2 == 0:
                    # 红色
                    color = 0xFF0000
                else:
                    # 橙色
                    color = 0xFFA500

                gui.circle((x[t, i][0]*scale/gui_length, x[t, i][1]*scale/gui_width), color, pixel_radius)

            gui.show()
    

    
if __name__ == '__main__':
    forward(visualize=rendering)
    print('Simulation Finished.')
