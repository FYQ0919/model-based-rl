# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import taichi as ti

from enum import Enum

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

class A1:
    class B:
        f = 1
        def func(self):
            print("1")
    def __init__(self):
        self.b = self.B()
    def func(self):
        self.b.func()

class A2:
    class B:
        f = 1
        def func(self):
            print("1")
    def __init__(self):
        self.b = B()
    def func(self):
        self.b.func()
