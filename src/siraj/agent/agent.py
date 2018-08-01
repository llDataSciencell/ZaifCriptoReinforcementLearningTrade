import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.buy_inventory = []
        self.sell_inventory = []
        self.inventory = []#train.pyで使用
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def old_model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def _model(self):
        shp=np.array([0 for i in range(10)])
        input1 = keras.layers.Input(shape=(None,self.state_size), name="in1")
        x1 = keras.layers.Dense(30, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(None,self.state_size), name="in2")
        x2 = keras.layers.Dense(30, activation='relu')(input2)
        input3 = keras.layers.Input(shape=(None,self.state_size), name="in3")
        x3 = keras.layers.Dense(30, activation='relu')(input3)
        input4 = keras.layers.Input(shape=(None,self.state_size), name="in4")
        x4 = keras.layers.Dense(30, activation='relu')(input4)
        input5 = keras.layers.Input(shape=(None,self.state_size), name="in5")
        x5 = keras.layers.Dense(30, activation='relu')(input5)
        input6 = keras.layers.Input(shape=(None,self.state_size), name="in6")
        x6 = keras.layers.Dense(30, activation='relu')(input6)
        input7 = keras.layers.Input(shape=(None,self.state_size), name="in7")
        x7 = keras.layers.Dense(30, activation='relu')(input7)
        input8 = keras.layers.Input(shape=(None,self.state_size), name="in8")
        x8 = keras.layers.Dense(30, activation='relu')(input8)
        added = keras.layers.Add()([x1, x2, x3, x4, x5, x6,x7,x8])  # equivalent to added = keras.layers.add([x1, x2])
        dense_added = keras.layers.Dense(150)(added)
        out = keras.layers.Dense(self.action_size, activation="linear", name="output_Q")(dense_added)
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7, input8], outputs=[out])

        model.compile(loss={'output_Q': 'mean_absolute_error'},
                      loss_weights={'output_Q': 1},
                      optimizer=Adam(lr=0.001))
        return model

    def act(self, state,buy_sell_array):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(({"in1":np.array([[state[0]]]),
                                           "in2":np.array([[state[1]]]),
                                           "in3": np.array([[state[2]]]),
                                           "in4": np.array([[state[3]]]),
                                           "in5": np.array([[state[4]]]),
                                           "in6": np.array([[state[5]]]),
                                           "in7": np.array([[state[6]]]),
                                           "in8": np.array([[state[7]]]),
                                           "buy_sell": np.array([[buy_sell_array]])}))

        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done, buy_sell_array in mini_batch:
            target = reward
            if not done:
                #print(state)
                #print("STATE:")
                #print(np.array(state[0]))
                #print(np.array(state[1]))
                #print(np.array(state[2]))
                #print(np.array(state[3]))
                #print(np.array(state[4]))
                #print(np.array(state[5]))
                #print("SHAPE:"+str(np.array(state[0]).shape))
                #print(np.array([0 for i in range(10)]).shape)
                target = reward + self.gamma * np.amax(self.model.predict({"in1": np.array([[state[0]]]),
                                                                           "in2": np.array([[state[1]]]),
                                                                           "in3": np.array([[state[2]]]),
                                                                           "in4": np.array([[state[3]]]),
                                                                           "in5": np.array([[state[4]]]),
                                                                           "in6": np.array([[state[5]]]),
                                                                           "in7": np.array([[state[6]]]),
                                                                           "in8": np.array([[state[7]]]),
                                                                           "buy_sell": np.array([[buy_sell_array]])
                                                                           }))

            target_f = self.model.predict({"in1":np.array([[state[0]]]),
                                           "in2":np.array([[state[1]]]),
                                           "in3": np.array([[state[2]]]),
                                           "in4": np.array([[state[3]]]),
                                           "in5": np.array([[state[4]]]),
                                           "in6": np.array([[state[5]]]),
                                           "in7": np.array([[state[6]]]),
                                           "in8": np.array([[state[7]]]),
                                           "buy_sell": np.array([[buy_sell_array]])})
            #print(target_f)
            target_f[0][0][action] = target

            self.model.fit({"in1":np.array([[state[0]]]),
                            "in2":np.array([[state[1]]]),
                            "in3": np.array([[state[2]]]),
                            "in4": np.array([[state[3]]]),
                            "in5": np.array([[state[4]]]),
                            "in6": np.array([[state[5]]]),
                            "in7": np.array([[state[6]]]),
                            "in8": np.array([[state[7]]]),
                            "buy_sell": np.array([[buy_sell_array]])}, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
