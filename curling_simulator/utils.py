# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import taichi as ti

from enum import Enum
from .config import *

class Player(Enum):
    A = 0 
    B = 1 

player_color = [0xFF0000, 0xFFA500]
def get_color(player:Player):
    """获取player的冰壶颜色"""
    return player_color[player.value]

def opponent(player:Player) -> Player:
    """获取对手信息"""
    return Player.B if player == Player.A else Player.A

def simple_reacale_decision(n_env:int,  
            # 原始的上下界
            old_low:np.ndarray,
            old_high:np.ndarray,
            # 新上下界
            low:float=-1.0,
            high:float=1.0,
            # 将 (1,4) shape变化为 (4,)
            flatten:bool = False,):
    """一个简单冰壶智能体,用于帮助RL agent生成一些简单的动作, 
    用于并行环境 + scale action"""
    pos = np.random.uniform(-0.03,0.03, size=(n_env,1))
    vx = np.random.uniform(7.4,7.6, size=(n_env,1))
    vy = np.random.uniform(-0.4,0.4, size=(n_env,1))
    w = np.random.uniform(vy/30, vy / 20, size=(n_env,1))
    w[vy<0] *= -1
    action = np.concatenate([pos, w, vx, vy], axis=1)
    # rescale
    action = (action-old_low)*(high-low)/(old_high-old_low) + low
    if n_env == 1 and flatten:
        return action.flatten()
    else:
        return action

class GameBoard(object):
    """记录比赛的双方分数"""
    def __init__(self,scores=None):
        ## 两队分数
        self.scores = np.empty((n_ends, 2), dtype=int)
        ## 重置分数板
        self.reset(scores)
    
    def __str__(self):
        """将score信息转化为str"""
        pass
    
    def summarize(self):
        """汇总当前所有分数，给出比赛结果"""
        res = self.scores.sum(axis=0)
        win_player = np.argmax(res)
        return res, Player(win_player)
    
    def write_win_score(self, curr_end:int, win_player:Player, win_score:int):
        """记录单局比赛的结果
        win_player 赢家, win_score 赢家分数
        """
        # 检查输入有效性
        assert curr_end >= 0
        assert 0 <= win_score <= n_curling_stones / 2
        # 检查curr_end比赛是否已记过分
        if self.scores[curr_end].any():
            print("[WARNING] game_board 该局比赛已记过分数，将覆盖")
        # 写入分数
        self.scores[curr_end, win_player.value] = win_score
    
    def write_score(self, curr_end:int, score:list or np.ndarray):
        """记录单局比赛的结果
        score: 维度(2,)的列表，比赛双方结果
        """
        # 检查输入有效性
        assert curr_end >= 0 and len(score) == 2
        assert np.min(score) >= 0 and np.max(score) <= n_curling_stones / 2
        # 检查curr_end比赛是否已记过分
        if self.scores[curr_end].any():
            print("[WARNING] game_board 该局比赛已记过分数，将覆盖")
        # 写入分数
        self.scores[curr_end] = score
    
    def reset(self, scores=None):
        """重置比分记录，或者指定分数"""
        if scores is not None:
            assert np.array(scores).shape == (n_ends, 2)
            self.scores = np.array(scores, dtype=int)
        else:
            self.scores = np.zeros((n_ends, 2), dtype=int)
        return 

