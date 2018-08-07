#https://github.com/keras-rl/keras-rl/blob/3dcd547f8f274f04fe11e813f52ceaed8987c90a/tests/rl/agents/test_dqn.py

import numpy as np
import gym

#from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Add,Flatten,Reshape
from keras.models import  Model
from rl.agents.dqn import DQNAgent
#from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.processors import MultiInputProcessor
from keras import backend as K
#K.set_image_dim_ordering('th')
ENV_NAME = 'FxEnv-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
len_data=env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
'''
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(90))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())
'''

def _model():
    #input1 = Input(shape=(2, 3))
    #input2 = Input(shape=(2, 4))
    #x = Concatenate()([input1, input2])
    #x = Flatten()(x)
    #x = Dense(2)(x)
    #model = Model(inputs=[input1, input2], outputs=x)


    state_size=20
    input_buy_sell = Input(shape=(1, 4), name="buy_sell")
    buy_sell = Dense(30, activation='relu')(input_buy_sell)
    input1 = Input(shape=(1, state_size), name="in1")
    x1 = Dense(30, activation='relu')(input1)
    input2 = Input(shape=(1, state_size), name="in2")
    x2 = Dense(30, activation='relu')(input2)
    input3 = Input(shape=(1, state_size), name="in3")
    x3 = Dense(30, activation='relu')(input3)
    input4 = Input(shape=(1, state_size), name="in4")
    x4 = Dense(30, activation='relu')(input4)
    input5 = Input(shape=(1, state_size), name="in5")
    x5 = Dense(30, activation='relu')(input5)
    input6 = Input(shape=(1, state_size), name="in6")
    x6 = Dense(30, activation='relu')(input6)
    input7 = Input(shape=(1, state_size), name="in7")
    x7 = Dense(30, activation='relu')(input7)
    input8 = Input(shape=(1, state_size), name="in8")
    x8 = Dense(30, activation='relu')(input8)

    added = Add()([buy_sell,x1, x2, x3, x4, x5, x6, x7, x8])  # equivalent to added = keras.layers.add([x1, x2])
    fl=Flatten()(added)
    dense_added = Dense(150)(fl)
    out = Dense(nb_actions, activation="linear",name="output_Q")(dense_added)
    #out=Reshape((-1,3),name="output_Q")(out_raw)
    #out=Flatten(name="output_Q")(out_raw)
    model = Model(inputs=[input_buy_sell,input1, input2, input3, input4, input5, input6, input7, input8], outputs=[out])

    model.compile(loss={'output_Q': 'mean_absolute_error'},
                  loss_weights={'output_Q': 1},
                  optimizer=Adam(lr=0.001))
    return model

model=_model()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.1) #BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
#processor = MultiInputProcessor(nb_inputs=9)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# MultiInputTestEnv([(3,), (4,)])
print(vars(dqn))
dqn.fit(env, nb_steps=len_data*5, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#dqn.test(env, nb_episodes=1, visualize=False)

