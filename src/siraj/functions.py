import numpy as np
import math
import os
import requests
import json, sys


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    print(vec)
    return vec


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
def getState(data, t, n):
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
def getStateLiveMode(price_array):
    res=[]
    for i in range(0,len(price_array)-1):
        res.append(sigmoid(int(price_array[i + 1] - price_array[i]) * 0.01))

    return res

def getStateFromCsvData(data,t,window_size):
    #tはidxに+1したもの。添字の都合。
    price300_sec_high=getStateLiveMode(data[t-window_size*int(300/60):t:int(300/60)])
    price3600_sec_high=getStateLiveMode(data[t-window_size*int(3600/60):t:int(3600/60)])
    price86400_sec_high=getStateLiveMode(data[t-window_size*int(86400/60):t:int(86400/60)])
    price300_sec_low=getStateLiveMode(data[t-window_size*int(300/60):t:int(300/60)])
    price3600_sec_low=getStateLiveMode(data[t-window_size*int(3600/60):t:int(3600/60)])
    price86400_sec_low=getStateLiveMode(data[t-window_size*int(86400/60):t:int(86400/60)])
    #print("price300:　　"+str(price300_sec_high))
    #print("data[始まり:終わり]　　　"+str(data[int(t-window_size*300/60):int(t)]))
    #print("data[t]"+str(data[t-1]))
    return price300_sec_high

def make_input_data(window_size):
    # ローソク足の時間を指定
    # periods = ["60","300"]

    # ローソク足取得
    # １日:86400 1時間:3600　1分 60
    #res3600 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=3600").text)
    res300 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=300").text)
    #res86400 = json.loads(requests.get("https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=86400").text)

    price300_sec_high = [res300["result"]["300"][-idx][2] for idx in range(1, window_size+1)]
    #price3600_sec_high=[res3600["result"]["3600"][-idx][2] for idx in range(1,window_size+1)]
    #price86400_sec_high = [res86400["result"]["86400"][-idx][2] for idx in range(1, window_size+1)]
    price300_sec_low = [res300["result"]["300"][-idx][3] for idx in range(1, window_size+1)]
    #price3600_sec_low=[res3600["result"]["3600"][-idx][3] for idx in range(1,window_size+1)]
    #price86400_sec_low = [res86400["result"]["86400"][-idx][3] for idx in range(1, window_size+1)]

    print(price300_sec_high)
    #必ずreturnでは複数の配列を返すこと。でないとtestcaseでエラーが出る。
    return getStateLiveMode(price300_sec_high),getStateLiveMode(price300_sec_low)#\
    '''
        ,getStateLiveMode(price3600_sec_high),\
           getStateLiveMode(price86400_sec_low),\
           getStateLiveMode(price3600_sec_low),getStateLiveMode(price86400_sec_low)
    '''
import unittest
from functions import *
window_size = 50
data=read_bitflyer_json()
class ObsForTest():
    def __init__(self):
        self.data = read_bitflyer_json()
        self.t=0
        self.window_size=50
class TestStringMethods(unittest.TestCase):

    def test_make_input1(self):
        self.assertEqual(len(make_input_data(window_size+1)[0]), 50)

    def test_Live_getState(self):
        idx=1000
        getStateFromCsvData(data, idx+1, window_size)
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
