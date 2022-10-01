# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import itertools

import taichi as ti
from enum import Enum
import math

from .config import *

@ti.data_oriented
class Simulation:
    def __init__(self,):
        # 冰壶shot仿真相关共享变量
        self.x = ti.Vector.field(n=2, shape=(max_steps, n_curling_stones), dtype=ti_real)
        self.v = ti.Vector.field(n=2, shape=(max_steps, n_curling_stones), dtype=ti_real)
        self.w = ti.field(shape=(n_curling_stones), dtype=ti_real)
        self.vas = ti.field(shape=(n_curling_stones), dtype=ti_int)
        # 仅供simulation使用的共享变量
        self.x_inc = ti.Vector.field(n=2, shape=(max_steps, n_curling_stones), dtype=ti_real)
        self.impulse = ti.Vector.field(n=2, shape=(max_steps, n_curling_stones), dtype=ti_real)
        self.f_ = ti.field(shape=(n_curling_stones), dtype=ti_real) # 平均作用力
        self.theta = ti.field(shape = (n_curling_stones), dtype = ti_real) # 初始角度
        # stone_pair
        self.set_stone_pairs()
        # 设置GUI窗口
        self.gui = None

        
    def set_stone_pairs(self):
        """设置球之间的pairs, 方便后续计算碰撞的并行"""
        # 创建field
        pair_num = n_curling_stones * (n_curling_stones-1)
        self.stone_pairs = ti.Vector.field(n=2, shape=(pair_num,), dtype=ti_int)
        # 生成球号之间的pairs,并填入field.
        itr = itertools.product(range(n_curling_stones), range(n_curling_stones))
        pairs = [pair for pair in itr if pair[0] != pair[1]]
        self.stone_pairs.from_numpy(np.array(pairs, dtype=np.int32))

    def open_gui(self,):
        """打开GUI"""
        self.gui = ti.GUI("Curling", (gui_length, gui_width), background_color=0xEEEEEE)

    def close_gui(self,):
        """关闭GUI"""
        self.gui.close()
        self.gui = None

    @ti.func
    def collide_pair(self, t, i, j):
        imp = ti.Vector([0.0, 0.0])
        x_inc_contrib = ti.Vector([0.0, 0.0])
        if i != j:
            dist = (self.x[t, i] + dt * self.v[t, i]) - (self.x[t, j] + dt * self.v[t, j])
            dist_norm = dist.norm()
            rela_v = self.v[t, i] - self.v[t, j]
            if dist_norm < 2 * radius:

                dir = ti.Vector.normalized(dist)
                projected_v = dir.dot(rela_v)

                if projected_v < 0:
                    imp = -(1 + elasticity) * 0.5 * projected_v * dir
                    toi = (dist_norm - 2 * radius) / ti.min(
                        -1e-3, projected_v)  # Time of impact
                    x_inc_contrib = ti.min(toi - dt, 0) * imp
        self.x_inc[t + 1, i] += x_inc_contrib
        self.impulse[t + 1, i] += imp

    @ti.kernel
    def collide(self, t: ti.i32):

        for idx in self.stone_pairs:
            # 获取两球的id
            i, j = self.stone_pairs[idx]
            # 检测是否有效
            if self.vas[i] != STATE_VALID or self.vas[j] != STATE_VALID:
                continue
            # 计算碰撞
            self.collide_pair(t, i, j)


    @ti.kernel
    def init_shot_sim(self,):
        """每次仿真shot之前执行此步骤"""
        for i in range(n_curling_stones):

            self.theta[i] = ti.atan2(self.v[0, i][1], self.v[0, i][0])

    @ti.kernel
    def time_integrate(self, t: ti.i32):
        for i in range(n_curling_stones):
            # 检测球是否有效
            if self.vas[i] != STATE_VALID:
                self.x[t, i] = self.x[t-1, i]
                continue
            v_total = ti.max(self.v[t - 1, i].norm() - miu * g * dt * (1 - f_), 0.0)  # 速度衰减
            # 计算 wp
            w_p = (v_total / r_p)
            self.theta[i] -= dt * f_ * w_p * self.w[i] # 速度夹角
            self.v[t, i] = [v_total * ti.cos(self.theta[i]),
                    v_total * ti.sin(self.theta[i])] + self.impulse[t, i]  # 速度

            self.x[t, i] = self.x[t - 1, i] + dt * (1 - f_)* self.v[t, i] + self.x_inc[t, i] # 位置

    @ti.func
    def out_of_bound(self, t:ti_int, id:ti_int)->ti_int:
        """
        判断是否出了边线或后卫线 1--出界; 0--未出界
        壶身完全处在后卫线之后 或 冰壶触及边线，均为出界
        """
        return self.x[t,id][0] + radius > l1 + 1.5 * l2 + 2 * l3 + l4 or \
        self.x[t,id][1] - radius < 0 or self.x[t,id][1] + radius > width

    @ti.func
    def before_hog_line(self, t:ti_int, id:ti_int)->ti_int:
        """
        判断壶身是否在hog line前 1--是; 0--否
        有一点压线就认为在line前
        """
        return self.x[t, id][0] - radius < l1 + l2 + l3 + l4

    @ti.func
    def move_away(self, t:ti_int, id:ti_int):
        """移走id冰壶"""
        self.x[t,id] = [- (id / 4 * 3 * radius) - 2 * radius,
                        - (id % 4 * 3 * radius) - 2 * radius]
        self.v[t,id] = [0, 0]
        self.x_inc[t,id] = [0, 0]
        self.vas[id] = STATE_INVALID
            
    def draw_ground(self):
        """绘制冰壶场地"""
        # Tee
        self.gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r4*scale), color=0x000080)
        self.gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r3*scale), color=0xEEEEEE)
        self.gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r2*scale), color=0xDC143C)
        self.gui.circle(((l1+l2)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r1*scale), color=0xEEEEEE)

        self.gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r4*scale), color=0x000080)
        self.gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r3*scale), color=0xEEEEEE)
        self.gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r2*scale), color=0xDC143C)
        self.gui.circle(((l1+l2+l3+l4+l3)*scale/gui_length, (width/2)*scale/gui_width), radius=int(r1*scale), color=0xEEEEEE)

        # Tee line
        self.gui.line(((l1+l2)*scale/gui_length, 0), ((l1+l2)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)
        self.gui.line(((l1+l2+l3+l4+l3)*scale/gui_length, 0), ((l1+l2+l3+l4+l3)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)

        # 后卫线
        self.gui.line(((l1+l2-r4)*scale/gui_length, 0), ((l1+l2-r4)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)
        self.gui.line(((l1+l2+l3+l4+l3+r4)*scale/gui_length, 0), ((l1+l2+l3+l4+l3+r4)*scale/gui_length, width*scale/gui_width), radius=1, color=0xFFB6C1)
        
        # Centre line
        self.gui.line((l1*scale/gui_length, width/2*scale/gui_width), ((length-l1)*scale/gui_length, width/2*scale/gui_width), radius=1, color=0xCD5C5C)

        # Hog line
        self.gui.line(((l1+l2+l3)*scale/gui_length, 0), ((l1+l2+l3)*scale/gui_length, width*scale/gui_width), radius=4, color=0xDC143C)
        self.gui.line(((l1+l2+l3+l4)*scale/gui_length, 0), ((l1+l2+l3+l4)*scale/gui_length, width*scale/gui_width), radius=4, color=0xDC143C)

        # Hack
        self.gui.line((l1*scale/gui_length, 0.5-hack_length/2*scale/gui_width), (l1*scale/gui_length, 0.5+hack_length/2*scale/gui_width), radius=5, color=0x000000)
        self.gui.line(((length-l1)*scale/gui_length, 0.5-hack_length/2*scale/gui_width), ((length-l1)*scale/gui_length, 0.5+hack_length/2*scale/gui_width), radius=5, color=0x000000)

    def draw_stones(self, pos, t:int=0, draw_num:bool=True, draw_vel:bool=True):
        """绘制冰壶
        pos: 冰壶位置,两种选择:
            1. x (steps, 12, 2), 这时候需要传入时间 t 
            2. env.stones_pos (12,2), 这时候t=0
        绘制样式: ```编号, [横轴速度,纵轴速度], w值*100```
        """
        if type(pos) == np.ndarray:
            pos = pos.reshape(1, n_curling_stones, 2)
        pixel_radius = int(radius*scale)+1
        for i in range(n_curling_stones):
            if i % 2 == 0:
                color = 0xFF0000  # 红色, 先手
            else:
                color = 0xFFA500  # 橙色, 后手
            # 绘制冰壶
            self.gui.circle([pos[t, i][0]*scale/gui_length, pos[t, i][1]*scale/gui_width], color, pixel_radius)
            # 绘制冰壶的速度
            vel = ""
            if draw_vel and self.v[t,i][0] + self.v[t,i][0] > static_velocity:
                self.gui.arrow(orig=[pos[t, i][0]*scale/gui_length, pos[t, i][1]*scale/gui_width],
                            direction=self.v[t,i]*scale/gui_length*0.4,
                            radius=2.0,
                            color=0x878a37
                            )
                vel += f",[{self.v[t,i][0]:.1f},{self.v[t,i][1]:.2f}],{self.w[i]:.1f}"
            # 绘制冰壶编号
            if draw_num:
                self.gui.text(f"{i}{vel}", 
                        [(pos[t, i][0])*scale/gui_length, (pos[t, i][1]+radius/2)*scale/gui_width], 
                        font_size=15,
                        color=0x6e6e6e)

    # 定义用于静止检测和出界检测的共享变量
    moving_num = ti.field(shape = (), dtype = ti_int)
    remove_num = ti.field(shape = (), dtype = ti_int)

    @ti.kernel
    def detect_stones(self, t:ti_int):
        """出界检测和静止检测"""
        self.moving_num[None] = 0  # 当前运动的球的数量
        self.remove_num[None] = 0  # 当前移走的球的数量 
        for i in range(n_curling_stones):
            # 判断球 i 是否有效
            if self.vas[i] != STATE_VALID:
                continue 
            # 判断是否出界
            res = self.out_of_bound(t, i)
            # 如果出界, 即刻移除
            if res == 1:
                self.remove_num[None] += 1
                self.move_away(t, i)
                continue
            # 判断静止
            is_static = (self.v[t,i][0] + self.v[t,i][1] < static_velocity) and all(self.impulse[t,i] == 0)
            if is_static == 1:
                # 检测到静止, 检测是否过hog line
                res = self.before_hog_line(t, i)
                if res == 1:
                    self.remove_num[None] += 1
                    self.move_away(t, i)
            else: # 球仍在运动
                self.moving_num[None] += 1
        return 

    def forward(self, curr_shot, render=True, interval=vis_interval, detect=True, wait_time=0.000): #TODO: 用于看效果时调整速率
        """仿真单个shot, 直到shot结束
        curr_shot: Int  env.curr_shot
        render:    Bool 是否渲染
        interval:  Int  GUI绘制的间隔
        detect:    Bool 是否做出界检测和静止检测
        """
        for t in range(1, max_steps):
            ## 计算碰撞
            self.collide(t - 1)
            self.time_integrate(t)
            ## 绘制GUI
            if (t + 1) % interval == 0 and render:
                self.draw_ground()
                self.draw_stones(self.x, t)
                self.gui.show()
                time.sleep(wait_time)
            ## 每过一定时间做一次出界检测和静止检测   
            if t % 15 != 0 or not detect:
                continue
            ## 检测出界和静止球
            self.detect_stones(t)

            # 五壶保护下有球被移走
            if curr_shot < 5 and self.remove_num[None] > 0:
                # print(f"检测到五壶保护内有球被移除, t={t}")
                return False, "FIVE_STONE_PROTECT", t
            ## 检测到所有球静止
            if self.moving_num[None] == 0:
                # print(f"检测到所有球静止, t={t}")
                return True, "OK", t
        # 达到 max_steps, 仍有球在运动, 返回 END_NOT_STATIC 代码   
        return True, "END_NOT_STATIC", -1,

