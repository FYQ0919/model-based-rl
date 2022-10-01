# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import math
import taichi as ti
import cv2

from .coordinate_extraction import get_position
# import torch

from . import curling_sim
from enum import Enum
from .config import *  # 重要的参数
from .utils import *   # 辅助函数与类


class CurlingSimuOneEndEnv(object):
    """用于仿真单局(End)比赛"""
    
    # taichi共享变量的numpy模板
    x_np = np.zeros((max_steps, n_curling_stones, 2), dtype=np.float32)
    
    def __init__(self, render=False):
        ## 开启一个simulation环境
        self.sim = curling_sim.Simulation()
        ## 当局先手队
        self.first_player = None
        ## 当前击球的队           
        self.curr_player = None
        ## 冰壶位置与有效性状态
        self.stones_pos = np.empty((n_curling_stones, 2), dtype=np.float32)
        self.stones_valid = np.empty((n_curling_stones,), dtype=np.int32)
        ## 游戏轮数
        self.curr_shot = None   
        ## 设置GUI
        self.if_render=render
        if self.if_render:
            self.sim.open_gui()
        # ## 初始化比赛场景
        # self.reset_simu(first_player)
    
    def reset_simu(self, first_player:Player=Player.A, render=None, bgr_img=None):
        """初始化比赛/重置比赛
        first_player: 此局(End)先手队员,默认为Player.A
        render: None表示维持当前reder状态, True/False表示打开/关闭render
        """
        ## 维护render
        if (render is not None) and (render != self.if_render):
            self.if_render = render
            if render:
                self.sim.open_gui()
            else:
                self.sim.close_gui()
        
        if bgr_img is None:
            ## 初始化队员信息
            self.curr_player = first_player
            self.first_player = first_player
            ## 初始化轮数, 初始击打第三个球, 有固定壶
            self.curr_shot = 2
            ## 初始化冰壶位置
            # 放置两个固定壶
            self.stones_pos[0] = [l1+l2+l3+l4+l3-r4-2.286, width/2]
            self.stones_pos[1] = [l1+l2+l3+l4+l3+r2-radius, width/2]
            # 其他壶放置并排放置
            for i in range(1, int(n_curling_stones/2)):
                self.stones_pos[i*2]   = [0.70, i*0.60]
                self.stones_pos[i*2+1] = [0.15, i*0.60]
            ## 设置冰壶的有效性状态
            self.stones_valid[0] = STATE_VALID
            self.stones_valid[1] = STATE_VALID
            self.stones_valid[2:] = STATE_READY
        else:
            ## 壶的位置
            stones_pos_r, stones_pos_y = get_position(bgr_img)
            for i in range(stones_pos_r.shape[0]):
                self.stones_pos[i*2] = stones_pos_r[i]
            for i in range(stones_pos_y.shape[0]):
                self.stones_pos[i*2+1] = stones_pos_y[i]
            ## 轮数
            red_num = stones_pos_r.shape[0]
            yellow_num = stones_pos_y.shape[0]
            self.curr_shot = red_num + yellow_num
            ## 初始化队员信息
            self.first_player = first_player
            if(self.curr_shot%2 == 0):
                self.curr_player = first_player
            else:
                self.curr_player = opponent(first_player)
            ## 其它壶并排放置
            for i in range(int(self.curr_shot/2), int(n_curling_stones/2)):
                if i*2>=self.curr_shot:
                    self.stones_pos[i*2]   = [0.70, i*0.60]
                    self.stones_pos[i*2 + 1] = [0.15, i*0.60]
                else:
                    self.stones_pos[i*2 + 1] = [0.15, i*0.60]
            ## 壶的有效性状态
            for i in range(0, self.curr_shot):
                if(self.stones_pos[i,0] > valid_pos_max[0]):
                    self.stones_valid[i] = STATE_INVALID
                else:
                    self.stones_valid[i] = STATE_VALID
            self.stones_valid[self.curr_shot:] = STATE_READY
        
        # print(self.stones_pos)
        # print(self.stones_valid)

        
        ## 更新GUI
        self._refresh_gui()
        return True
        
    def done(self):
        """是否结束"""
        isdone = self.curr_shot >= n_curling_stones
        return isdone
    
    def shot_vec(self, 
                vec:list or np.ndarray or torch.tensor,
                player:Player or int=None):
        """用于解析上层policy输出的action vector"""
        return self.shot(pos_x = vec[0], 
                         pos_y = vec[1], 
                         w = vec[2],
                         vel=vec[3:5], 
                         player=player)
    
    def shot_vec_nox(self, 
                vec:list or np.ndarray or torch.tensor,
                player:Player or int=None,
                pos_x:float=2.3):
        """用于解析上层policy输出的action vector, vec不包含pos_x"""
        return self.shot(pos_x = pos_x, 
                         pos_y = vec[0], 
                         w = vec[1],
                         vel=vec[2:4], 
                         player=player)
      
    def shot_vec_noy(self, 
                vec:list or np.ndarray or torch.tensor,
                player:Player or int=None,
                pos_y:float=0.0):
        """用于解析上层policy输出的action vector, vec不包含pos_y"""
        return self.shot(pos_x = vec[0], 
                         pos_y = pos_y, 
                         w = vec[1],
                         vel=vec[2:4], 
                         player=player)
    
    def shot_vec_auto(self, 
                vec:list or np.ndarray or torch.tensor,
                player:Player or int=None,
                pos_x:float=None,
                pos_y:float=None):
        """用于解析上层policy输出的action vector, vec可能不包含pos_x和pos_y
        ATTENTION! 务必保证: 给定 pos_x 和 pos_y 时 vec中不再包含这两个pos信息!!
        """
        pos_x = pos_x if pos_x is not None else vec[0]
        pos_y = pos_y if pos_y is not None else vec[-4]
        return self.shot(pos_x = pos_x, 
                         pos_y = pos_y, 
                         w = vec[-3],
                         vel=vec[-2:], 
                         player=player)
    
    def shot(self,
            pos_x:float=2.3,
            pos_y:float=0.0,
            w:float=0.0,
            vel:list or np.ndarray=[7.5,0.1], 
            player:Player or int=None):
        """
        投掷冰壶
        pos_x: float 控制出手位置 x 坐标 [0.0, 4.57]
        pos_x: float 控制出手位置 y 坐标 [-0.5, 0.5]
            出手位置坐标: [l1+l2+r4+pos_x, width*(pos_y+0.5)]
        w: float 自旋, 负为逆时针, 正为顺时针  [-0.15, 0.15]
        vel: [linear_velocity, lateral_velocity] 出手速度
            - 横轴 [7.0,9.0]
            - 纵轴 [-2.0,2.0]
        player: A or B 当前击球的player
        """
        # # DEBUG
        # print(pos_x, pos_y, w, vel, player)
        ## 对输入进行检查
        assert self.done() == False
        assert player == self.curr_player
        ## 将当前状态和击打信息，输入到仿真变量中
        # 冰壶位置
        self.x_np[0] = self.stones_pos
        self.sim.x.from_numpy(self.x_np)
        self.sim.x[0, self.curr_shot] = [l1+l2+r4+pos_x, 
                                         width*(pos_y+0.5)]
        # 冰壶初速度
        self.sim.v.fill(0.0)
        self.sim.v[0, self.curr_shot] = vel
        # 冰壶自旋
        self.sim.w.fill(0.0)
        self.sim.w[self.curr_shot] = w
        # 其他参数
        self.sim.x_inc.fill(0.0)
        self.sim.impulse.fill(0.0)
        # 冰壶有效性
        self.sim.vas.from_numpy(self.stones_valid)
        self.sim.vas[self.curr_shot] = STATE_VALID
        ## 开始仿真
        self.sim.init_shot_sim()
        res, res_str, end_time,  = self.sim.forward(curr_shot=self.curr_shot, render=self.if_render)

        ## 判断是否出现异常，比如触犯五壶保护规则
        # 异常情况处理
        if res_str == 'OK':
            # 正常情况, 记录shot结果
            self.stones_pos = self.sim.x.to_numpy()[end_time]
            self.stones_valid = self.sim.vas.to_numpy()
        elif res_str == 'FIVE_STONE_PROTECT':
            # print("本次击球无效, 原因:", res_str)
            # 五壶保护情况下当前shot的球invalid
            self.stones_valid[self.curr_shot] = STATE_INVALID
            # 把新打的球放进垃圾场
            self.stones_pos[self.curr_shot] = [- (self.curr_shot / 4 * 3 * radius) - 2 * radius, 
                                               - (self.curr_shot % 4 * 3 * radius) - 2 * radius]   
        elif res_str == 'END_NOT_STATIC':
            # forward结束仍有球运动, 记录shot结果
            self.stones_pos = self.sim.x.to_numpy()[end_time]
            self.stones_valid = self.sim.vas.to_numpy()
            print(f"[WARNING] forward结束时仍然有球运动. [{pos_x:.3f}, {pos_y:.3f}, {w:.5f}, {vel[0]:.3f}, {vel[1]:.4f}]")      
        else:
            raise NotImplementedError("sim.forward返回未知状态码.")
        # 轮换到下一个队击球
        self._next_shot()
        return res, res_str
    
    def _next_shot(self):
        """
        转换到下一个shot
        返回值表示游戏是否结束
        """
        self.curr_shot += 1
        self.curr_player = opponent(self.curr_player)
        # 更新GUI
        self._refresh_gui()
        return 
    
    def score(self):
        """计算得分
        返回值 [Player.A的score, Player.B的score]"""
        ## 计算所有球与中心点的距离
        dis = np.linalg.norm(self.stones_pos - center_pos, axis=1)
        ## 判断球的有效性: 在场，并且在大本营内
        valid = self.stones_valid == STATE_VALID
        dis[~valid] = np.inf
        dis[dis>r4 + radius] = np.inf
        ## 取出双方的球的信息
        # 取出Player.A的球index, 先手为0
        pa_idx = 1 - int(Player.A == self.first_player)
        stones_a, stones_b = dis[pa_idx::2], dis[(1-pa_idx)::2]    
        min_a, min_b = stones_a.min(), stones_b.min()
        ## 计算双方分数
        if min_a < min_b:
            score_a = np.sum(stones_a < min_b)
            return [score_a, 0]
        else:
            score_b = np.sum(stones_b < min_a)
            return [0, score_b]
    
    def score_zero_sum(self):
        """零和博弈版本的score"""
        score = self.score()
        if score[0] < score[1]:
            score[0] = score[1] * -1
        else:
            score[1] = score[0] * -1
        return score
    
    def observe_stones_pos(self, 
                           player:Player=None,
                           normalize:bool=True, 
                           fill_value:float=invalid_fill_value):
        """获取player队伍球的位置
        player: 获取哪个球队的球信息
        normalize: 是否归一化
        fill_value: 无效球的值
        """
        # 确定player的先后手, 先手为 0
        p_idx = 1 - int(player == self.first_player)
        # 取出 player 的球
        stones_pos = self.stones_pos[p_idx::2].copy()
        stones_valid = self.stones_valid[p_idx::2].copy()
        # 球位置normalization
        if normalize:
            stones_pos = (stones_pos - valid_pos_min) / valid_pos_range
        # 无效球信息填充
        stones_pos[stones_valid != STATE_VALID, :] = fill_value
        # 按照x轴进行排序
        pos_vec = stones_pos[stones_pos[:,0].argsort()].astype(np.float32)
        # print(pos_vec)
        return pos_vec
    
    def observe(self, *args,**kwargs):
        """通用observe接口"""
        raise NotImplementedError("请在gym环境中定义observation函数!")
    
    def save_game(self):
        """保存env信息"""
        pass
    
    def load(self, stones):
        """加载冰壶位置信息"""
        
        pass
    
    def _refresh_gui(self):
        """重置GUI, 辅助函数"""
        if not self.if_render:
            return 
        # print("refresh")
        self.sim.draw_ground()
        self.sim.draw_stones(self.stones_pos, draw_vel=False)
        self.sim.gui.show()

class CurlingSimuEnv(object):
    """用于仿真整场比赛, 8个end"""
    def __init__(self,):
        ## 创建记分板
        self.game_board = GameBoard()

        pass
    
    def _next_end(self):
        """下一场比赛"""
        pass
    
    def done(self):
        pass
    
    def reset(self):
        pass
    

if __name__ == '__main__':
    pass
    path = '../coordinate_extraction/test_games/01/13/14.png'
    img = cv2.imread(path)
    env = CurlingSimuOneEndEnv(render=True)
    env.reset_simu(Player.A, bgr_img=img)
    # env.action(0.0, -0.02, [7.5,-0.50], env.curr_player)
    # env.action(0.0, 0.0, [8.5,1.00], env.curr_player)
    # env = CurlingSimuEnv()