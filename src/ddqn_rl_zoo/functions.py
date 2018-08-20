import numpy as np
import math
import os
import requests
import json, sys
import matplotlib.pyplot as plt

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def make_inventory_array(buy_inventory,sell_inventory,max_inventory,current_price):
    #常にinventoryが長さ50の配列になるように
    if len(buy_inventory) > 0 and len(sell_inventory) > 0:
        # current_price - inv
        buy_inventory=[int(current_price - inv) for inv in buy_inventory]
        buy_inv = buy_inventory+[0 for i in range(50-len(buy_inventory))]
        # inv - current_price
        sell_inventory=[int(inv - current_price) for inv in sell_inventory]
        sell_inv = sell_inventory+[0 for i in range(50-len(sell_inventory))]
    elif len(buy_inventory) > 0:
        buy_inventory=[int(current_price - inv) for inv in buy_inventory]
        buy_inv = buy_inventory+[0 for i in range(50-len(buy_inventory))]
        sell_inv = [0 for i in range(0,50)]
    elif len(sell_inventory) > 0:
        sell_inventory=[int(inv - current_price) for inv in sell_inventory]
        buy_inv = sell_inventory+[0 for i in range(50-len(sell_inventory))]
        sell_inv = [0 for i in range(0,50)]
    else:
        buy_inv=[0 for i in range(50)]
        sell_inv=[0 for i in range(50)]
    return buy_inv,sell_inv

def read_bitflyer_json():
    import csv
    history_data = []
    import csv
    with open(os.environ['HOME'] + '/bitcoin/bitflyerJPY_convert.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)  # ヘッダーを読み飛ばしたい時
        for row in reader:
            history_data.append(float(row[1]))
        # print(float(row[1]))
        return history_data


# returns the sigmoid
def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

'''
>>> ccc[:-100:-2]
[299, 297, 295, 293, 291, 289, 287, 285, 283, 281, 279, 277, 275, 273, 271, 269, 267, 265, 263, 261, 259, 257, 255, 253, 251, 249, 247, 245, 243, 241, 239, 237, 235, 233, 231, 229, 227, 225, 223, 221, 219, 217, 215, 213, 211, 209, 207, 205, 203, 201]
>>>

'''

# returns an an n-day state representation ending at time t
# 入力データを作る際も、価格ごとの差分をとってシグモイドに入れている。
def getState(data, idx, window_size):
    n=window_size+1
    t=idx+1
    #t(idx)に+1したのは、
    #aaa[100:200:10] =>[100, 110, 120, 130, 140, 150, 160, 170, 180, 190]で200番目の数字がこぼれてしまうから。
    #idxを+1しておくことで、内包表記が簡単になる。
    #>>> aaa[100:201:10] => [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(int(block[i + 1] - block[i]) * 0.01))

    return np.array([res])
def getStateBySigmoid(price_array):
    #入力->出力で１つ配列が短くなる。
    res=[]
    for i in range(0,len(price_array)-1):
        res.append(sigmoid(int(price_array[i + 1] - price_array[i]) * 0.01))

    return res
def calc_low(data,idx,window_size,one_tick_sec_term):
    if idx <= window_size * (one_tick_sec_term/60):
        #TODO modify
        return [0 for i in range(0,window_size)]
    low_price=[]
    low=float('inf')
    for i in range(idx,idx-(window_size+1)*int(one_tick_sec_term/60),-1):
        if data[i] < low:
            low=data[i]
        if len(low_price) >= window_size:
            #print("data:   ")
            #print(data[idx - (window_size + 1) * int(one_tick_sec_term / 60):idx + 1])
            low_price.reverse()
            #print("low_price array:")
            #print(low_price)
            return low_price
        if i % int(one_tick_sec_term/60) == 0:
            low_price.append(low)
            low = float('inf')


def calc_high(data,idx,window_size,one_tick_sec_term):

    if idx <= window_size * (one_tick_sec_term/60):
        #TODO modify
        return [0 for i in range(0,window_size)]
    high_price=[]
    high=-float('inf')
    for i in range(idx,idx-(window_size+1)*int(one_tick_sec_term/60),-1):
        if data[i] > high:
            high=data[i]
        if len(high_price) >= window_size:
            #print("data:   ")
            #print(data[idx - (window_size + 1) * int(one_tick_sec_term / 60):idx + 1])
            high_price.reverse()
            return high_price
        if i % int(one_tick_sec_term/60) == 0:
            high_price.append(high)
            high = -float('inf')

    high_price.reverse()
    return high_price

def calc_60(data,idx,window_size,one_tick_sec_term):
    zeros=[]
    if idx <= window_size:
        zeros = [0 for i in range(0,window_size-idx-1)]
    price=[]
    for i in range(idx,idx-(window_size+1)*int(one_tick_sec_term/60),-1):
        if len(price) >= window_size-len(zeros):
            price.reverse()
            return zeros+price
        price.append(data[i])
    price.reverse()
    return zeros+price

def getStateFromCsvData(data,idx,window_size):
    t=idx+1
    #tはidxに+1したもの。添字の都合。
    #print("getStateFromCsvData"+str(calc_high(data,idx,window_size+1,300)))
    price60_sec_high = getStateBySigmoid(calc_60(data, idx, window_size + 1, 60))
    price300_sec_high=getStateBySigmoid(calc_high(data,idx,window_size+1,300))
    price3600_sec_high=getStateBySigmoid(calc_high(data,idx,window_size+1,3600))
    price86400_sec_high=getStateBySigmoid(calc_high(data,idx,window_size+1,86400))
    price300_sec_low=getStateBySigmoid(calc_low(data,idx,window_size+1,300))
    price3600_sec_low=getStateBySigmoid(calc_low(data,idx,window_size+1,3600))
    price86400_sec_low=getStateBySigmoid(calc_low(data,idx,window_size+1,86400))
    #print("price300:　　"+str(price300_sec_high))
    #print("data[] 300/60　　　"+str(data[int(t-window_size*300/60)-1:int(t):int(300/60)]))
    #print("data[] 3600/60    "+str(data[t - window_size * int(3600 / 60) - 1:t:int(3600 / 60)]))
    #print("data[idx]"+str(data[t-1]))
    #print("data[idx]" + str(data[t-50:t]))
    return [np.array(price60_sec_high),np.array(price300_sec_high),np.array(price3600_sec_high),np.array(price86400_sec_high),np.array(price300_sec_low), np.array(price3600_sec_low), np.array(price86400_sec_low)]

def make_input_data(window_size):
    # ローソク足の時間を指定
    # periods = ["60","300"]

    # ローソク足取得
    # １日:86400 1時間:3600　1分 60
    #res3600 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=3600").text)
    res300 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=300").text)
    #res86400 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=86400").text)

    price300_sec_high = [res300["result"]["300"][-idx][2] for idx in range(1, window_size+2)]
    #price3600_sec_high=[res3600["result"]["3600"][-idx][2] for idx in range(1,window_size+2)]
    #price86400_sec_high = [res86400["result"]["86400"][-idx][2] for idx in range(1, window_size+2)]
    price300_sec_low = [res300["result"]["300"][-idx][3] for idx in range(1, window_size+2)]
    #price3600_sec_low=[res3600["result"]["3600"][-idx][3] for idx in range(1,window_size+2)]
    #price86400_sec_low = [res86400["result"]["86400"][-idx][3] for idx in range(1, window_size+2)]

    #print(price300_sec_high)
    #必ずreturnでは複数の配列を返すこと。でないとtestcaseでエラーが出る。
    return getStateBySigmoid(price300_sec_high),getStateBySigmoid(price300_sec_low)#\
    '''
        ,getStateBySigmoid(price3600_sec_high),\
           getStateBySigmoid(price86400_sec_low),\
           getStateBySigmoid(price3600_sec_low),getStateBySigmoid(price86400_sec_low)
    '''
price_history=[]
trade_history=[]
def update_trading_view(current_price, action):
        price_history.append(current_price)
        trade_history.append(action)


def draw_trading_view():
        global price_history,trade_history
        data, date = np.array(price_history), np.array([idx for idx in range(0, len(price_history))])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(date, data)#,marker='o'
        ax.plot()

        for num in range(0,len(price_history)):
            if trade_history[num] == 1:
                plt.scatter(date[num], data[num], marker="^", color="red")
            elif trade_history[num] == 2:
                plt.scatter(date[num], data[num], marker="^", color="red")
            elif trade_history[num] == 3:
                plt.scatter(date[num],data[num], marker="v", color="green")
            elif trade_history[num] == 4:
                plt.scatter(date[num],data[num], marker="v", color="green")

        ax.set_title("Cripto Price")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price[$]")
        plt.grid(fig)
        plt.show(fig)
        price_history=[]
        trade_history=[]

import unittest
window_size = 20
data=read_bitflyer_json()

class TestStringMethods(unittest.TestCase):
    #calc_high,calc_lowはwindow_size+1の長さの配列を返却

    #Liveはwindow_sizeぴったり
    def test_make_input1(self):
        self.assertEqual(len(make_input_data(window_size)[0]), window_size)

    def test_getStateCsv(self):
        idx=1000
        self.assertEqual(len(getStateFromCsvData(data, idx, window_size)[0]),window_size)

    def test_calc_high_low1(self):
        self.assertEqual(len(calc_low(data,1000,window_size,300)),window_size)
    def test_calc_high_low2(self):
        self.assertEqual(len(calc_low(data,1001,window_size,300)),window_size)
    def test_calc_high_low3(self):
        self.assertEqual(len(calc_low(data,1002,window_size,300)),window_size)
    def test_calc_high_low4(self):
        self.assertEqual(len(calc_low(data,1003,window_size,300)),window_size)
    def test_calc_high_low5(self):
        self.assertEqual(len(calc_low(data,1004,window_size,300)),window_size)
    def test_calc_high_low6(self):
        self.assertEqual(len(calc_low(data,1005,window_size,300)),window_size)
    def test_calc_high_low7(self):
        self.assertEqual(len(calc_low(data,1006,window_size,300)),window_size)
    def test_calc_high_low8(self):
        self.assertEqual(len(calc_low(data,1007,window_size,300)),window_size)
    def test_calc_high1(self):
        self.assertEqual(len(calc_high(data,1000,window_size,300)),window_size)
    def test_calc_high2(self):
        self.assertEqual(len(calc_high(data,1001,window_size,300)),window_size)
    def test_calc_high3(self):
        self.assertEqual(len(calc_high(data,1002,window_size,300)),window_size)
    def test_calc_high4(self):
        self.assertEqual(len(calc_high(data,1003,window_size,300)),window_size)
    def test_calc_high5(self):
        self.assertEqual(len(calc_high(data,1004,window_size,300)),window_size)
    def test_calc_high6(self):
        self.assertEqual(len(calc_high(data,1005,window_size,300)),window_size)
    def test_calc_high7(self):
        self.assertEqual(len(calc_high(data,1006,window_size,300)),window_size)
    def test_calc_high8(self):
        self.assertEqual(len(calc_high(data,1007,window_size,300)),window_size)

    def test_calc_high_low1(self):
        self.assertEqual(len(calc_low(data,1000,window_size,86400)),window_size)
    def test_calc_high_low2(self):
        self.assertEqual(len(calc_low(data,1001,window_size,86400)),window_size)
    def test_calc_high_low3(self):
        self.assertEqual(len(calc_low(data,1002,window_size,86400)),window_size)
    def test_calc_high_low4(self):
        self.assertEqual(len(calc_low(data,1003,window_size,86400)),window_size)
    def test_calc_high_low5(self):
        self.assertEqual(len(calc_low(data,1004,window_size,86400)),window_size)
    def test_calc_high_low6(self):
        self.assertEqual(len(calc_low(data,1005,window_size,300)),window_size)
    def test_calc_high_low7(self):
        self.assertEqual(len(calc_low(data,1006,window_size,300)),window_size)
    def test_calc_high_low8(self):
        self.assertEqual(len(calc_low(data,1007,window_size,300)),window_size)
    def test_calc_high1(self):
        self.assertEqual(len(calc_high(data,1000,window_size,300)),window_size)
    def test_calc_high2(self):
        self.assertEqual(len(calc_high(data,1001,window_size,300)),window_size)
    def test_calc_high3(self):
        self.assertEqual(len(calc_high(data,1002,window_size,300)),window_size)
    def test_calc_high4(self):
        self.assertEqual(len(calc_high(data,1003,window_size,300)),window_size)
    def test_calc_high5(self):
        self.assertEqual(len(calc_high(data,1004,window_size,300)),window_size)
    def test_calc_high6(self):
        self.assertEqual(len(calc_high(data,1005,window_size,300)),window_size)
    def test_calc_high7(self):
        self.assertEqual(len(calc_high(data,1006,window_size,300)),window_size)
    def test_calc_high8(self):
        self.assertEqual(len(calc_high(data,1007,window_size,300)),window_size)

    def test_calc_high2_1(self):
        self.assertEqual(len(calc_high(data, 1008, window_size, 300)), window_size)

    def test_calc_high2_2(self):
        self.assertEqual(len(calc_high(data, 1009, window_size, 300)), window_size)

    def test_calc_high2_3(self):
        self.assertEqual(len(calc_high(data, 1010, window_size, 300)), window_size)

    def test_calc_high2_4(self):
        self.assertEqual(len(calc_high(data, 1011, window_size, 300)), window_size)

    def test_calc_high2_5(self):
        self.assertEqual(len(calc_high(data, 1012, window_size, 300)), window_size)

    def test_calc_high2_6(self):
        self.assertEqual(len(calc_high(data, 1013, window_size, 300)), window_size)

    def test_calc_high2_7(self):
        self.assertEqual(len(calc_high(data, 1014, window_size, 300)), window_size)

    def test_calc_high2_8(self):
        self.assertEqual(len(calc_high(data, 1015, window_size, 300)), window_size)


    def test_calc_high2_9(self):
        self.assertEqual(len(calc_high(data, 0, window_size, 300)), window_size)
    def test_calc_high2_10(self):
        self.assertEqual(len(calc_high(data, 1, window_size, 300)), window_size)

    def test_high_price(self):
        self.assertEqual(calc_high([0 if idx % 3 == 0 else 100 for idx in range(0,3000)],1007,window_size,300)[10],100)

    def test_getState(self):
        self.assertEqual(len(getStateBySigmoid(calc_high(data, 1000, window_size+1, 300))),window_size)

    def test_getState2(self):
        self.assertEqual(len(getStateBySigmoid(calc_high(data, 1, window_size+1, 300))),window_size)
    def test_calc_60(self):
        #print(data[30:51])
        #rint(calc_60(data, 50, window_size+1, 60))
        self.assertEqual(len(calc_60(data, 50, window_size+1, 60)),len(data[30:51]))
    def test_calc_60_2(self):
        self.assertEqual(calc_60(data, 50, window_size+1, 60),data[50-(window_size):51])
    def test_calc_60_3(self):
        #print(data[0:10])
        #print(calc_60(data,10,window_size+1,60))
        self.assertEqual(calc_60(data, 10, window_size+1, 60)[10:20],data[0:10])
    '''
    def test_make_input2(self):
        self.assertEqual(len(make_input_data(window_size)[1]), 50)
    def test_make_input3(self):
        self.assertEqual(len(make_input_data(window_size)[2]), 50)
    def test_make_input4(self):
        self.assertEqual(len(make_input_data(window_size)[3]), 50)
    def test_make_input5(self):
        self.assertEqual(len(make_input_data(window_size)[4]), 50)
    def test_make_input6(self):
        self.assertEqual(len(make_input_data(window_size)[5]), 50)
    def test_make_input6(self):
        self.assertEqual(len(make_input_data(window_size)), 6)
    def test_make_input6(self):
        print(data[1900:2000])
        self.assertEqual(len(getState_from_csvdata(data, 2000, window_size)),window_size)
    '''

    '''
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    '''
if __name__ == '__main__':
    unittest.main()
