#coding: utf-8
import chainer
#import chainer.functions as F
#import chainer.links as L
#import chainerrl
from chainerrl.agents import a3c
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
trade=TradeClass()
price_data = trade.read_bitflyer_json()
print("price_data idx 0-10"+str(price_data[0:10]))
print("price_data idx last 10"+str(price_data[-1]))

'''
def getDataPoloniex():
    polo = poloniex.Poloniex()
    polo.timeout = 10
    chartUSDT_BTC = polo.returnChartData('USDT_BTC', period=300, start=time.time() - 1440 * 60 * 500, end=time.time())  # 1440(min)*60(sec)=DAY
    tmpDate = [chartUSDT_BTC[i]['date'] for i in range(len(chartUSDT_BTC))]
    date = [datetime.datetime.fromtimestamp(tmpDate[i]) for i in range(len(tmpDate))]
    data = [float(chartUSDT_BTC[i]['open']) for i in range(len(chartUSDT_BTC))]
    return date, data
'''

#time_date, price_data = getDataPoloniex()
obs_size = 200#shape#env.observation_space.shape[0]
n_actions=3

training_set=copy.copy(price_data)
X_train = []
y_train = []
for i in range(obs_size, len(training_set)-1001):
    #X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_train.append(training_set[i - obs_size:i])
    y_train.append(training_set[i])


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(100,100)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)



model = A3CFFSoftmax(ndim_obs=obs_size, n_actions=n_actions)
#opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=0.01, alpha=0.98)
opt=chainer.optimizers.Adam(eps=1e-3)
opt.setup(model)

# Set the discount factor that discounts future rewards.

class RandomActor:
    def __init__(self):
        pass
    def random_action_func(self):
        return random.randint(0,2)

ra = RandomActor()

def phi(obs):
    return obs.astype(np.float32)

agent = a3c.A3C(model, opt, t_max=3, gamma=0.95, beta=1e-2, phi=phi,act_deterministically=True)

#action_value=chainerrl.action_value.ActionValue()

def env_execute(action,current_price,next_price,cripto_amount,usdt_amount):

    return reward

def buy_simple(pred,money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = money * 0.8
        money -= spend *1.0005
        if money < 0.0:
            return first_money,first_ethereum,first_total_money

        ethereum += float(spend / current_price)
        total_money = money + ethereum * current_price

        return money, ethereum, total_money

def sell_simple(pred,money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = ethereum * 0.8
        ethereum -= spend * 1.0005
        if ethereum < 0.0:
            return first_money,first_ethereum,first_total_money

        money += float(spend * current_price)
        total_money = money + float(ethereum * current_price)

        return money, ethereum, total_money
def pass_simple(pred,money,ethereum,total_money,current_price):
    total_money = money + float(ethereum * current_price)
    return money,ethereum,total_money
try:
  agent.load('polo_agent')
except:
    print("Agent load failed")

#トレーニング
#放っておいてtotal_priceが上がることもある。今のプログラムだと放っておいても値段変わらない
for i in range(0,3):
    reward=0

    price = y_train
    money = 300
    before_money = money
    ethereum = 0.01
    total_money = money + np.float64(price[0] * ethereum)
    first_total_money = total_money
    pass_count=0
    for idx in range(0, len(price)):
                current_price = X_train[idx][-1]
                action = agent.act_and_train(np.array(X_train[idx],dtype='f'), reward)#idx+1が重要。
                #Qmax=agent.evaluate_actions(action)

                Qmax=1
                pass_reward=0
                if action == 0:
                    #print("buy")
                    money, ethereum, total_money = buy_simple(Qmax,money, ethereum, total_money, current_price)
                elif action == 1:
                    #print("sell")
                    money, ethereum, total_money = sell_simple(Qmax,money, ethereum, total_money, current_price)
                else:
                    #print("PASS")
                    money, ethereum, total_money = pass_simple(Qmax,money, ethereum, total_money, current_price)
                    pass_reward=0.00#0.01 is default
                    pass_count+=1

                reward = total_money - before_money+pass_reward
                before_money = total_money

                if idx % 100 == 1:
                    print("action:" + str(action))
                    print("FINAL" + str(total_money))
                    print("100回中passは"+str(pass_count)+"回")
                    pass_count=0
    #agent.stop_episode_and_train(X_train[-1], reward, True)

    # Save an agent to the 'agent' directory
    agent.save('chainerRLAgent')
    print("START MONEY" + str(first_total_money))

#テスト
for i in range(0,1):
    reward=0

    price = y_train
    money = 300
    before_money = money
    ethereum = 0.01
    total_money = money + np.float64(price[0] * ethereum)
    first_total_money = total_money
    for idx in range(0, len(price)):
                print(i)
                current_price = price[idx]
                action = agent.act(np.array(X_train[idx],dtype='f'))#idx+1が重要。
                Qmax=1.0#.evaluate_actions(action)


                if action == 0:
                    print("buy")
                    money, ethereum, total_money = buy_simple(Qmax,money, ethereum, total_money, current_price)
                elif action == 1:
                    print("sell")
                    money, ethereum, total_money = sell_simple(Qmax,money, ethereum, total_money, current_price)
                else:
                    print("PASS")

                reward = total_money - before_money
                before_money = total_money

                print("FINAL" + str(total_money))
    # Save an agent to the 'agent' directory
    agent.save('chainerRLAgent')
    print("START MONEY" + str(first_total_money))

