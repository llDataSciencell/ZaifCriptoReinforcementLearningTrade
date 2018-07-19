import copy
from trade_class import TradeClass
import gym
import numpy as np
class FxEnv(gym.Env):
    def __init__(self):
        self.price_idx=0
        self.trade = TradeClass()
        training_set = self.trade.read_bitflyer_json()
        print("price_data idx 0-10" + str(training_set[0:10]))
        print("price_data idx last 10" + str(training_set[-1]))

        self.input_len = 400
        self.n_actions = 3
        self.obs_size = self.input_len + self.n_actions  #env.observation_space.shape[0]

        
        self.X_train = []
        self.y_train = []
        for i in range(self.input_len, len(training_set) - 1001):
            # X_train.append(np.flipud(training_set_scaled[i-60:i]))
            self.X_train.append(training_set[i - self.input_len:i])
            self.y_train.append(training_set[i])

        self.price = self.y_train
        self.money = 300
        self.before_money = copy.deepcopy(self.money)
        self.cripto = 0.1
        self.total_money = self.money + np.float64(self.price[0] * self.cripto)
        self.first_total_money = self.total_money
        self.pass_count = 0
        self.buy_sell_count = 0  # buy+ sell -
        self.pass_renzoku_count = 0
        self.buy_sell_fee=0.00001

    def _reset(self):
        self.price_idx=0
        return self.X_train[self.price_idx]
    def _seed(self,seed=None):
        pass
    def buy_simple(self,money, cripto, total_money, current_price):
            first_money, first_cripto, first_total_money = money, cripto, total_money
            spend = money * 0.1
            money -= spend * (1+self.buy_sell_fee)
            if money <= 0.0:
                return first_money,first_cripto,first_total_money

            cripto += float(spend / current_price)
            total_money = money + cripto * current_price

            return money, cripto, total_money

    def sell_simple(self,money, cripto, total_money, current_price):
        first_money, first_cripto, first_total_money = money, cripto, total_money
        spend = cripto * 0.1
        cripto -= spend * (1+self.buy_sell_fee)
        if cripto <= 0.0:
            return first_money,first_cripto,first_total_money

        money += float(spend * current_price)
        total_money = money + float(cripto * current_price)

        return money, cripto, total_money
    def pass_simple(self,money,cripto,total_money,current_price):
        total_money = money + float(cripto * current_price)
        return money,cripto,total_money
    def _step(self, action):
        self.price_idx+=1

        current_price = self.X_train[self.price_idx][-1]
        buy_sell_num_array = [1.0, 0.0, self.buy_sell_count] if self.buy_sell_count >= 1 else [0.0, 1.0, self.buy_sell_count]

        self.trade.update_trading_view(current_price, action)

        pass_reward = 0
        if action == 0:
            print("buy")
            self.buy_sell_count += 1
            self.money, self.cripto, self.total_money = self.buy_simple(self.money, self.cripto, self.total_money, current_price)
        elif action == 1:
            print("sell")
            self.buy_sell_count -= 1
            self.money, self.cripto,self.total_money = self.sell_simple(self.money, self.cripto, self.total_money, current_price)
        else:
            print("PASS")
            self.money, self.cripto, self.total_money = self.pass_simple(self.money, self.cripto, self.total_money, current_price)
            pass_reward = 0.0  # -0.001)#0.01 is default
            self.pass_count += 1

        reward = self.total_money - self.before_money + pass_reward
        if self.buy_sell_count >= 5 and action == 0:
            print("buy_sell" + str(self.buy_sell_count) + "回　action==" + str(action))
            reward -= (float(abs(self.buy_sell_count) ** 2))
            print(reward)
        elif self.buy_sell_count <= -5 and action == 1:
            print("buy_sell" + str(self.buy_sell_count) + "回　action==" + str(action))
            reward -= (float(abs(self.buy_sell_count) ** 2))
            print(reward)
        else:
            # reward 1.0がちょうど良い！
            reward += 1.1

        self.before_money = copy.deepcopy(self.total_money)

        if self.price_idx % 2000 == 1000:
            print("last action:" + str(action))
            print("TOTAL MONEY" + str(self.total_money))
            print("100回中passは" + str(self.pass_count) + "回")
            # print("100回中buy_sell_countは" + str(self.buy_sell_count) + "回")
            self.pass_count = 0
            self.trade.draw_trading_view()

        current_asset=[self.cripto,self.money]

        # obs, reward, done, infoを返す
        return [self.X_train[self.price_idx],buy_sell_num_array,current_asset],reward,False,None

