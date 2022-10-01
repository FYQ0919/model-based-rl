# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import taichi as ti

from enum import Enum

from .gym_env import *
from gym.wrappers import RescaleAction

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

"""测试env"""
# env = RescaleAction(CurlingSingleAgentGymEnv(render=False), min_action=-1.0, max_action=1.0)
# check_env(env)

"""建立并行env"""
env = make_vec_env(lambda: RescaleAction(CurlingSingleAgentGymEnv(render=True), min_action=-1.0, max_action=1.0),
                   n_envs=3, vec_env_cls=DummyVecEnv)  # change render to False when training !!

obs = env.reset()
obs, reward, done, info = env.step(np.random.rand(3,4))

"""
1. 建议包装scale action 
2. 使用DummyVecEnv, 而不是 SubprocVecEnv
3. action: (n_envs, 4) 每个环境分配动作, 或者  (4,) 所有的env都执行这个动作
4. done = True时, 环境会自动reset, 这时获得的obs不是结束的obs, 而是reset后的obs
"""

"""SubprocVecEnv无法正常工作"""
# env = make_vec_env(lambda: RescaleAction(CurlingSingleAgentGymEnv(render=True), min_action=-1.0, max_action=1.0),
#                    n_envs=3, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork')) # fork, forkserver, spawn 都不正常


