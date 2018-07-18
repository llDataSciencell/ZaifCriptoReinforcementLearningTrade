#coding: utf-8
import chainer
#import chainer.functions as F
#import chainer.links as L
import chainerrl
from chainerrl.agents import a3c
import chainer.links as L
import chainer.functions as F

#from chainerrl.action_value import DiscreteActionValue
#from chainerrl.action_value import QuadraticActionValue
#from chainerrl.optimizers import rmsprop_async

from chainerrl import links
from chainerrl import policies

import numpy as np
import random
import time

import poloniex
import datetime
import copy
from trade_class import TradeClass

try:
  agent.load('polo_agent')
except:
    print("Agent load failed")

import gym
import random

env = gym.make('FxEnv-v0')

Episodes = 1

obs = []

for _ in range(Episodes):
    observation = env.reset()
    done = False
    count = 0
    while not done:
        action = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
        observation, reward, done, info = env.step(action)
        obs = obs + [observation]
        # print observation,reward,done,info
        count += 1
        if done:
            print('reward:', reward)
            print('steps:', count)

