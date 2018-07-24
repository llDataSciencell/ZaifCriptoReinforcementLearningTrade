#coding: utf-8
import chainer
#import chainer.functions as F
#import chainer.links as L
import chainerrl
from chainerrl.agents import a3c
import chainer.links as L
import chainer.functions as F
import os, sys
print(os.getcwd())
sys.path.append(os.getcwd())
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

input_len=400
n_actions=3
obs_size = input_len+n_actions#shape#env.observation_space.shape[0]

training_set=copy.copy(price_data)
X_train = []
y_train = []
for i in range(input_len, len(training_set)-1001):
    #X_train.append(np.flipud(training_set_scaled[i-60:i]))
    X_train.append(training_set[i - input_len:i])
    y_train.append(training_set[i])


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=500):
        super(QFunction, self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))



model = QFunction(obs_size=obs_size, n_actions=n_actions)
#opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=0.01, alpha=0.98)
opt=chainer.optimizers.Adam(eps=1e-5)
opt.setup(model)

# Set the discount factor that discounts future rewards.
gamma = 0.95

class RandomActor:
    def __init__(self):
        pass
    def random_action_func(self):
        # 所持金を最大値にしたランダムを返すだけ
        return random.randint(0,2)

ra = RandomActor()

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=ra.random_action_func)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)


agent = chainerrl.agents.DoubleDQN(
    model, opt, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

def env_execute(action,current_price,next_price,cripto_amount,usdt_amount):

    return reward

buy_sell_fee = 0.0005
def buy_simple(money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = money * 0.1
        money -= spend * (1+buy_sell_fee)
        if money <= 0.0:
            return first_money,first_ethereum,first_total_money

        ethereum += float(spend / current_price)
        total_money = money + ethereum * current_price

        return money, ethereum, total_money

def sell_simple(money, ethereum, total_money, current_price):
        first_money, first_ethereum, first_total_money = money, ethereum, total_money
        spend = ethereum * 0.1
        ethereum -= spend * (1+buy_sell_fee)
        if ethereum <= 0.0:
            return first_money,first_ethereum,first_total_money

        money += float(spend * current_price)
        total_money = money + float(ethereum * current_price)

        return money, ethereum, total_money
def pass_simple(money,ethereum,total_money,current_price):
    total_money = money + float(ethereum * current_price)
    return money,ethereum,total_money
try:
  agent.load('chainerRLAgent')
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
    buy_sell_count=0#buy+ sell -
    pass_renzoku_count=0
    total_reward=0

    for idx in range(0, len(price)):
                current_price = X_train[idx][-1]
                buy_sell_num_flag=[1.0,0.0,abs(buy_sell_count)] if buy_sell_count >= 1 else [0.0,1.0,abs(buy_sell_count)]
                action = agent.act_and_train(np.array(X_train[idx]+buy_sell_num_flag,dtype='f'), reward)#idx+1が重要。

                print(agent.get_statistics())

                trade.update_trading_view(current_price, action)
                reward=0
                if action == 0:
                    print("buy")
                    buy_sell_count+=1
                    money, ethereum, total_money = buy_simple(money, ethereum, total_money, current_price)
                elif action == 1:
                    print("sell")
                    buy_sell_count-=1
                    money, ethereum, total_money = sell_simple(money, ethereum, total_money, current_price)
                elif action==2:
                    print("PASS")
                    money, ethereum, total_money = pass_simple(money, ethereum, total_money, current_price)
                    reward+=0.00000
                    pass_count+=1
                reward += 0.01 * (total_money - before_money)  # max(current_price-bought_price,0)##



                before_money = total_money

                if idx % 10000 == 500:
                        print("BEGGINING MONEY:"+str(first_total_money))
                        print("FINAL MONEY:" + str(total_money))
                        print("1000回中passは"+str(pass_count)+"回")
                        print("1000回終わった後のbuy_sell_countは" + str(buy_sell_count) + "回")
                        print("total_reward:"+str(total_reward))
                        pass_count=0
                        try:
                            trade.draw_trading_view()
                        except:
                            pass
                        agent.save('chainerRLAgent')

    #agent.stop_episode_and_train(X_train[-1], reward, True)

    # Save an agent to the 'agent' directory
    agent.save('chainerRLAgent')
    print("START MONEY" + str(first_total_money))
