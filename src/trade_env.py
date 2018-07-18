import copy
from trade_class import TradeClass

class FxEnv(gym.Env):
    def __init__(self):
        self.price_idx=0
        self.trade = TradeClass()
        price_data = trade.read_bitflyer_json()
        print("price_data idx 0-10" + str(price_data[0:10]))
        print("price_data idx last 10" + str(price_data[-1]))

        input_len = 400
        n_actions = 3
        obs_size = input_len + n_actions  # shape#env.observation_space.shape[0]

        training_set = copy.copy(price_data)
        X_train = []
        y_train = []
        for i in range(input_len, len(training_set) - 1001):
            # X_train.append(np.flipud(training_set_scaled[i-60:i]))
            X_train.append(training_set[i - input_len:i])
            y_train.append(training_set[i])

        price = y_train
        money = 300
        before_money = money
        ethereum = 0.01
        total_money = money + np.float64(price[0] * ethereum)
        first_total_money = total_money
        pass_count = 0
        buy_sell_count = 0  # buy+ sell -
        pass_renzoku_count = 0


    def _reset(self):
        self.price_idx=0
        return X_train[0]

    def _step(self, action):

        self.price_idx+=1
        reward=0

        current_price = X_train[self.price_idx][-1]
        buy_sell_num_flag = [1.0, 0.0, buy_sell_count] if buy_sell_count >= 1 else [0.0, 1.0, buy_sell_count]

        self.trade.update_trading_view(current_price, action)

        pass_reward = 0
        if action == 0:
            print("buy")
            buy_sell_count += 1
            money, ethereum, total_money = buy_simple(money, ethereum, total_money, current_price)
        elif action == 1:
            print("sell")
            buy_sell_count -= 1
            money, ethereum, total_money = sell_simple(money, ethereum, total_money, current_price)
        else:
            print("PASS")
            money, ethereum, total_money = pass_simple(money, ethereum, total_money, current_price)
            pass_reward = 0.0  # -0.001)#0.01 is default
            self.pass_count += 1

        reward = total_money - before_money + pass_reward
        if buy_sell_count >= 5 and action == 0:
            print("buy_sell" + str(buy_sell_count) + "回　action==" + str(action))
            reward -= (float(abs(buy_sell_count) ** 2))
            print(reward)
        elif buy_sell_count <= -5 and action == 1:
            print("buy_sell" + str(buy_sell_count) + "回　action==" + str(action))
            reward -= (float(abs(buy_sell_count) ** 2))
            print(reward)
        else:
            # reward 1.0がちょうど良い！
            reward += 1.1

        before_money = total_money

        if idx % 2000 == 1000:
            print("last action:" + str(action))
            print("TOTAL MONEY" + str(total_money))
            print("100回中passは" + str(pass_count) + "回")
            # print("100回中buy_sell_countは" + str(buy_sell_count) + "回")
            self.pass_count = 0
            trade.draw_trading_view()
            agent.save('chainerRLAgent')

        # obs, reward, done, infoを返す
        return X_train[price_idx],reward,False,None