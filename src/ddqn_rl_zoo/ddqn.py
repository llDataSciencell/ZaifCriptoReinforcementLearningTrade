import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Add
from keras.models import Model
from functions import *

EPISODES = 300

# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.input_stream_size=7
        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.buy_sell_len=2
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_ddqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        input1 = Input(shape=(None, self.state_size), name="in1")
        x1 = Dense(30, activation='relu')(input1)
        input2 = Input(shape=(None, self.state_size), name="in2")
        x2 = Dense(30, activation='relu')(input2)
        input3 = Input(shape=(None, self.state_size), name="in3")
        x3 = Dense(30, activation='relu')(input3)
        input4 = Input(shape=(None, self.state_size), name="in4")
        x4 = Dense(30, activation='relu')(input4)
        input5 = Input(shape=(None, self.state_size), name="in5")
        x5 = Dense(30, activation='relu')(input5)
        input6 = Input(shape=(None, self.state_size), name="in6")
        x6 = Dense(30, activation='relu')(input6)
        input7 = Input(shape=(None, self.state_size), name="in7")
        x7 = Dense(30, activation='relu')(input7)

        added = Add()(
            [x1, x2, x3, x4, x5, x6, x7])  # equivalent to added = keras.layers.add([x1, x2])
        dense_added = Dense(150)(added)
        out = Dense(self.action_size, activation="linear", name="output_Q")(dense_added)
        model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7],
                                   outputs=[out])

        model.compile(loss={'output_Q': 'mean_absolute_error'},
                      loss_weights={'output_Q': 1},
                      optimizer=Adam(lr=0.001))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state,buy_sell):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(({"in1":np.array([[state[0]]]),
                                           "in2":np.array([[state[1]]]),
                                           "in3": np.array([[state[2]]]),
                                           "in4": np.array([[state[3]]]),
                                           "in5": np.array([[state[4]]]),
                                           "in6": np.array([[state[5]]]),
                                           "in7": np.array([[state[6]]]),
                                           "buy_sell": np.array([[buy_sell]])}))
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done,buy_sell,next_buy_sell):
        self.memory.append((state, action, reward, next_state, done,buy_sell,next_buy_sell))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size,self.input_stream_size,self.state_size))
        update_target = np.zeros((batch_size,self.input_stream_size,self.state_size))
        update_input_buy_sell=np.zeros((batch_size,1,self.buy_sell_len))
        update_target_buy_sell=np.zeros((batch_size,1,self.buy_sell_len))
        action, reward, done = [], [], []

        for i in range(batch_size):
            #print(i)
            #print(update_input.shape)#(64, 20)
            #print(np.array(mini_batch).shape)#(64, 5)
            #print(mini_batch[i][0])
            #print(mini_batch[i])
            #print(update_input)
            #print(update_input[0])
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
            update_input_buy_sell[i]=mini_batch[i][5]
            update_target_buy_sell[i]=mini_batch[i][6]
        #TODO buy_sell_arrayを修正する
        # buy_sell_array = np.array([[0,0,0,0] for j in range(batch_size)])
        target = self.model.predict(({"in1":np.array([update_input[:,0]]),
                                           "in2":np.array([update_input[:,1]]),
                                           "in3": np.array([update_input[:,2]]),
                                           "in4": np.array([update_input[:,3]]),
                                           "in5": np.array([update_input[:,4]]),
                                           "in6": np.array([update_input[:,5]]),
                                           "in7": np.array([update_input[:,6]]),
                                           "buy_sell": np.array([update_input_buy_sell[:]])}))

        target_next = self.model.predict(({"in1":np.array([update_target[:,0]]),
                                           "in2":np.array([update_target[:,1]]),
                                           "in3": np.array([update_target[:,2]]),
                                           "in4": np.array([update_target[:,3]]),
                                           "in5": np.array([update_target[:,4]]),
                                           "in6": np.array([update_target[:,5]]),
                                           "in7": np.array([update_target[:,6]]),
                                           "buy_sell": np.array([update_target_buy_sell[:]])}))

        #self.model.predict(update_target)
        target_val = self.target_model.predict(({"in1":np.array([update_target[:,0]]),
                                           "in2":np.array([update_target[:,1]]),
                                           "in3": np.array([update_target[:,2]]),
                                           "in4": np.array([update_target[:,3]]),
                                           "in5": np.array([update_target[:,4]]),
                                           "in6": np.array([update_target[:,5]]),
                                           "in7": np.array([update_target[:,6]]),
                                           "buy_sell": np.array([update_target_buy_sell[:]])}))

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[0][i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[0][i])
                #print(i)#64
                #print("a:" + str(a))#a:0
                target[0][i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[0][i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        # update_input
        self.model.fit({"in1": np.array([update_target[:, 0]]),
                                           "in2": np.array([update_target[:, 1]]),
                                           "in3": np.array([update_target[:, 2]]),
                                           "in4": np.array([update_target[:, 3]]),
                                           "in5": np.array([update_target[:, 4]]),
                                           "in6": np.array([update_target[:, 5]]),
                                           "in7": np.array([update_target[:, 6]]),
                                           "buy_sell": np.array([update_target_buy_sell[:]])}, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":

    window_size, episode_count = int(20), int(1000)

    print("window_size:" + str(window_size))
    print("episode_count:" + str(episode_count))

    data = read_bitflyer_json()

    length_data = len(data) - 1
    action_size = 3
    input_len=20
    agent = DoubleDQNAgent(input_len, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        state = getStateFromCsvData(data, 0, window_size)
        total_profit = 0
        agent.buy_inventory = []
        agent.sell_inventory = []
        for idx in range(length_data):
            state = getStateFromCsvData(data, idx, window_size)

            # get action for the current state and go one step in environment
            # action =
            # next_state, reward, done, info = env.step(action)
            len_buy = len(agent.buy_inventory)
            len_sell = len(agent.sell_inventory)


            buy_sell = [len_buy, len_sell]
            #state.append(buy_sell_array)
            #TODO buy_sell_array_nextを設定する。
            action = agent.get_action(state,buy_sell)
            # TODO idx + 1出なくて良いか？　バグの可能性あり。
            next_state = getStateFromCsvData(data, idx + 1, window_size)
            next_buy_sell=[len_buy+1,len_sell-1] if action == 0 else [len_buy-1,len_sell+1]
            reward = 0

            if action == 1 and len(agent.sell_inventory) > 0:
                i = 0
                for i in range(0, int(len(agent.sell_inventory) / 10)):
                    sold_price = agent.sell_inventory.pop(0)
                    profit = sold_price - data[idx]
                    reward += profit  # max(profit, 0)
                    total_profit += profit
                    print("Buy(決済): " + formatPrice(data[idx]) + " | Profit: " + formatPrice(profit))
                #reward = reward / (i + 1)
            elif action == 1 and len(agent.buy_inventory) < 50:
                agent.buy_inventory.append(data[idx])
                print("Buy: " + formatPrice(data[idx]))
            elif action == 2 and len(agent.buy_inventory) > 0:
                i = 0
                for i in range(0, int(len(agent.buy_inventory) / 10)):
                    bought_price = agent.buy_inventory.pop(0)
                    profit = data[idx] - bought_price
                    reward += profit  # max(profit, 0)
                    total_profit += profit
                    print("Sell: " + formatPrice(data[idx]) + " | Profit: " + formatPrice(profit))
                #reward = reward / (i + 1)
            elif action == 2 and len(agent.sell_inventory) < 50:
                agent.sell_inventory.append(data[idx])
                print("Sell(空売り): " + formatPrice(data[idx]))
            reward = reward / 1000
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done,buy_sell,next_buy_sell)
            # every time step do the training
            agent.train_model()
            state = next_state

            if idx % 20 == 0:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("BUY SELL" + str(buy_sell))
                print("--------------------------------")
            if idx % 100000 == 0:
                agent.model.save_weights("./save_model/epoch"+str(e)+"_"+str(idx)+"ddqn.h5")
