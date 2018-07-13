#coding: utf-8
#tensorflowがWindowsに対応していないので、scikit-learn使うようにした。
#エラーは1000くらい。だいぶ正解率が高くなった。

import chainerrl
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
from statsmodels.tsa.seasonal import seasonal_decompose
import poloniex
import datetime
import copy
from trade_class import TradeClass

INPUT_LEN = 100
SEQUENTIAL_NUM = 10

def create_data():
    trade=TradeClass()
    # time_date, price_data = trade.getDataPoloniex()
    num_history_data = trade.read_bitflyer_json()

    X=[]
    for idx in range(0,len(num_history_data)-INPUT_LEN-SEQUENTIAL_NUM):
        tmp=[num_history_data[idx+list_idx] for list_idx in range(0,INPUT_LEN)]
        X.append(tmp)


    #次の入力データXの最新のデータを、Xの教師Yとする。
    Y=[]
    tmp=[]
    for i in range(INPUT_LEN,len(num_history_data)-SEQUENTIAL_NUM):
        for j in range(0,SEQUENTIAL_NUM):
            tmp.append(float(num_history_data[i+j]))
        Y.append(tmp)
        tmp=[]
    print(Y[0])
    print("len(Y)"+str(len(Y)))
    print("len(X))"+str(len(X)))
    #XとYの配列の長さが等しくなるように、Xの要素を一つ削除する 。


    return X, Y

if __name__ == "__main__":
    X_raw, y = create_data()

    X = X_raw.astype(np.float32)

    try:
        chainerrl.Agent.agent.load('polo_agent')
    except:
        print("Agent load failed")

    print("length of DATA:"+str(len(X)))

