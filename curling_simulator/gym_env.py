# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import taichi as ti
from typing import TypeVar, Iterable, Tuple, Union

from .curling_sim import *
from enum import Enum
from .config import *  # 重要的参数
from .utils import *  # 辅助函数与类
import cv2

from .curling_env import *
import gym
from gym import spaces
from gym.utils import seeding

"""
TODO: 
- 设置图片observation

"""

"""一些常量, 方便debug使用"""
RENDER = False


class CurlingTwoAgentGymEnv_v0(gym.Env, CurlingSimuOneEndEnv):
    """gym环境: 两个智能体对打 v0 版本 """
    """出手坐标仅可控制x的值"""
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        print(f"[Gym] {self.__doc__}")
        # 初始化仿真环境
        gym.Env.__init__(self, )
        CurlingSimuOneEndEnv.__init__(self, render)
        # 设置obs, action等space
        self._elapsed_steps = 0
        self.observation_space = spaces.Box(low=np.inf * -1,
                                            high=np.inf,
                                            shape=network_input_size,
                                            dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([0.5,-0.05,7.3,-0.5], dtype=np.float32),
        #                               high=np.array([4.5,0.05,8.4,0.5], dtype=np.float32),
        #                               shape=(4,),
        #                               dtype=np.float32)
        self.action_space = spaces.Discrete(363)
        self.initialize_img = None

    def observe(self, player: Player):
        """observe接口"""
        """ATTENTION: 为了(可能)减小 muzero dynamic 模块学习难度, 
            obs中的双方球的pos先后顺序不再随着obs的player而变动"""
        # # 获取两队的球的位置信息
        # pos_vec = np.concatenate([
        #     self.observe_stones_pos(Player.A),
        #     self.observe_stones_pos(Player.B)
        # ])
        # # 获取当前shot的轮数
        # curr_turn_vec = np.array([(self.curr_shot - 2) // 2], dtype=np.float32)
        # # 先后手信息, 算法可以确定是哪一个agent
        # is_first_vec = np.array([player.value], dtype=np.float32)
        # # 合并信息

        # obs = np.concatenate([
        #     pos_vec.flatten(),
        #     curr_turn_vec,
        #     is_first_vec
        # ])

        # return obs.astype(np.float32)

        # obs = self.feature_img(self.curr_player)
        obs = self.simplified_feature_img(self.curr_player)
        return obs
    
    def simplified_feature_img(self, player: Player):
        def distance(stone):
            stone = np.resize(stone, (1, 2))
            return np.linalg.norm(center_pos-stone)

        # 先后手信息, 算法可以确定是哪一个agent
        is_first_vec = player.value
        feat = np.zeros(network_input_size, dtype=np.float32)
        
        if is_first_vec == 0:#curr Player A
            self_stones = self.observe_stones_pos(Player.A)
            oppo_stones = self.observe_stones_pos(Player.B)
        else:
            self_stones = self.observe_stones_pos(Player.B)
            oppo_stones = self.observe_stones_pos(Player.A)
            
        for i in range(self_stones.shape[0]):
            if(self_stones[i,0] != invalid_fill_value):
                h, w = self.position2point(self_stones[i])#all
                feat[0,h,w] = 1
                
                dis = distance(self_stones[i])
                if dis <= r4:#in house
                    h, w = self.position2point(self_stones[i])
                    feat[2,h,w] = 1
                    
        for i in range(oppo_stones.shape[0]):
            if(oppo_stones[i,0] != invalid_fill_value):
                h, w = self.position2point(oppo_stones[i])#all
                feat[1,h,w] = 1
                
                dis = distance(oppo_stones[i])#in house
                if dis <= r4:
                    h, w = self.position2point(oppo_stones[i])
                    feat[3,h,w] = 1
        
        return feat
        
        


    def feature_img(self, player: Player):
        def distance(stone):
            stone = np.resize(stone, (1, 2))
            return np.linalg.norm(center_pos-stone)

        # 先后手信息, 算法可以确定是哪一个agent
        is_first_vec = player.value
        
        feat = np.zeros(network_input_size, dtype=np.float32)
        feat[2,:,:] = 1
        #ones
        feat[3,:,:] = 1

        if is_first_vec == 0:#curr Player A
            A_stones = self.observe_stones_pos(Player.A)
            B_stones = self.observe_stones_pos(Player.B)
        else:
            A_stones = self.observe_stones_pos(Player.B)
            B_stones = self.observe_stones_pos(Player.A)

        #stone color
        for i in range(A_stones.shape[0]):
            if(A_stones[i,0] != invalid_fill_value):
                h, w = self.position2point(A_stones[i])
                feat[0,h,w] = 1
                feat[2,h,w] = 0
        for i in range(B_stones.shape[0]):
            if(B_stones[i,0] != invalid_fill_value):
                h, w = self.position2point(B_stones[i])
                feat[1,h,w] = 1
                feat[2,h,w] = 0


        #turn num
        # 获取当前shot的轮数
        curr_turn_vec = np.array([self.curr_shot // 2], dtype=np.float32) # -2?
        if(curr_turn_vec==0):
            feat[4 + is_first_vec*4,:,:] = 1
        elif(curr_turn_vec==1):
            if(is_first_vec==0):
                feat[4,:,:] = 1
            else:
                feat[9,:,:] = 1
        elif(curr_turn_vec==2 or curr_turn_vec==3 or curr_turn_vec==4 or curr_turn_vec==5):
            feat[5 + is_first_vec*4,:,:] = 1
        elif(curr_turn_vec==6):
            feat[6 + is_first_vec*4:,:] = 1   
        elif(curr_turn_vec==7):
            feat[7 + is_first_vec*4:,:] = 1 

        #in house
        for i in range(A_stones.shape[0]):
            if(A_stones[i,0] != -1.0):
                dis = distance(A_stones[i])
                if dis <= r4:
                    h, w = self.position2point(A_stones[i])
                    feat[12,h,w] = 1
        for i in range(B_stones.shape[0]):
            if(B_stones[i,0] != -1.0):
                dis = distance(B_stones[i])
                if dis <= r4:
                    h, w = self.position2point(B_stones[i])
                    feat[12,h,w] = 1

        #order to tee
        A_valid = []
        B_valid = []
        for i in range(A_stones.shape[0]):
            if(A_stones[i,0] != -1.0):
                A_valid.append(A_stones[i])
        for i in range(B_stones.shape[0]):
            if(B_stones[i,0] != -1.0):
                B_valid.append(B_stones[i])
        total_valid = A_valid + B_valid

        A_valid.sort(key = distance)
        B_valid.sort(key = distance)
        total_valid.sort(key = distance)

        # for i in range(len(A_valid)):
        #     h, w = self.position2point(A_valid[i])
        #     if i<4:
        #         feat[13 + i,h,w] = 1
        #     else:
        #         feat[16,h,w] = 1
        
        # for i in range(len(B_valid)):
        #     h, w = self.position2point(B_valid[i])
        #     if i<4:
        #         feat[17 + i,h,w] = 1
        #     else:
        #         feat[20,h,w] = 1

        for i in range(len(total_valid)):
            h, w = self.position2point(total_valid[i])
            if i<8:
                feat[13 + i,h,w] = 1
            else:
                feat[20,h,w] = 1

        if np.sum(feat)>=-1e10:
            pass
        else:
            print('feat wrong')
            for i in range(21):
                print(np.sum(feat[i]))
                print(feat[i])      
        return feat

    def position2point(self, position):
        # print(position)
        point = np.zeros((2, 1), dtype=np.int64)
        point[0] = (1-position[0])*network_input_size[0]
        point[1] = position[1] * network_input_size[1]

        return point

    def to_play(self):
        """to_play函数示例, 待确认是否有bug..."""
        return self.curr_player.value

    def step(self, action):
        """action (1,) index"""
        result = None
        obs = self.observe(self.curr_player)

        # 执行动作
        posx = 0
        w = [-1, 0, 1]
        vx = [2.2 + 0.03 * i for i in range(11)]
        vy_1 = [0.05 + 0.006 * i for i in range(11)]
        vy_2 = [-0.04 + 0.008 * i for i in range(11)]
        real_w = w[int(action // 121) % 3]
        if real_w < 0:
            real_vy = -(abs(vy_1[int(action) % 11]))
        elif real_w > 0:
            real_vy = (abs(vy_1[int(action) % 11]))
        else:
            real_vy = np.clip(vy_2[int(action) % 11], -0.02, 0.02)

        real_vx = vx[int(action // 11) % 11]

        if self.curr_shot <= 5:
            real_vx = np.clip(real_vx, a_min=0, a_max=2.3)

        real_action = [posx, real_w, real_vx, real_vy]

        # print(real_action)

        res, res_str = self.shot_vec_noy(vec=real_action, player=self.curr_player)

        # 获取观测和done信号
        obs = self.observe(self.curr_player)
        done = self.done()
        # 计算 reward

        if done:
            # 游戏规则得分, 得分赋予给后手 Player.B
            rew = self.score_zero_sum()[Player.B.value] * 10

            if rew > 0:
                result = "player 2 wins"
            elif rew < 0:
                result = "player 1 wins"
            else:
                result = "Tie"
        else:
            rew = 0.0

        # 其他奖励得分
        # p_idx = 1 - int(self.curr_player == self.first_player)
        # dis = np.linalg.norm(self.stones_pos[p_idx::2] - center_pos, axis=1)
        # rew += (2.5 - 1.01 * dis[dis <= r4 + radius]).sum()

        if res_str == 'FIVE_STONE_PROTECT':
            rew -= 1

        self._elapsed_steps += 1
        return obs, rew, done, {"result": result}

#?
    def reset(self, render: bool = None):
        # 初始化仿真环境, 确定先后手
        CurlingSimuOneEndEnv.reset_simu(self,
                                        first_player=Player.A,
                                        render=render,
                                        bgr_img=self.initialize_img)
        # 获取 obs
        obs = self.observe(self.curr_player)
        self._elapsed_steps = 0
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        return None

    def close(self):
        return None


if __name__ == '__main__':
    pass
    env = CurlingTwoAgentGymEnv_v0(render=True)
    # for i in range(50,300):
    #     obs = env.reset(render=True)
    path = '../coordinate_extraction/test_games/01/13/13.png'
    img = cv2.imread(path)
    obs = env.reset(render=True, bgr_img=img)
    obs, reward, done, info = env.step(0)
    # import pdb; pdb.set_trace()
    # envs = []
    # for _ in range(5):
    #     env = CurlingSingleAgentGymEnv()
    #     env.reset(first=None, render=True)
    #     envs.append(env)
    #     input(f"i={_}, input something. ")
