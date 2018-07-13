#coding: utf-8

import pickle
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import sys
global trade_zaif
from zaifapi.impl import *
from trade_class import TradeClass
import requests
import traceback
from keras.models import load_model
trade_zaif = ZaifTradeApi('47af66e6-6b29-4c9e-a538-1777858515fb','c4dc4c30-3271-4877-9cc8-871a5a2de8ba')
leverage_zaif = ZaifLeverageTradeApi('47af66e6-6b29-4c9e-a538-1777858515fb','c4dc4c30-3271-4877-9cc8-871a5a2de8ba')

public_zaif = ZaifPublicApi()

INPUT_LEN = 100#入力データの長さ
SEQUENTIAL_NUM = 10#出力ノードのノード数
DIVISION_NUM = 0.0001#小さな数字の方が良い
ONE_ORDER_AMOUNT_NUM = 0.0001#オーダの基本単位
model=load_model('keras_model.h5')


def convert_seasonal_trend(tmp):

    decomposition = seasonal_decompose(np.array(tmp, dtype=float), freq=7)

    tmp_trend = decomposition.trend
    tmp_trend[np.isnan(tmp_trend)] = 0

    tmp_seasonal = decomposition.seasonal
    tmp_seasonal[np.isnan(tmp_seasonal)] = 0

    return tmp ,tmp_trend, tmp_seasonal

FIRST=True
before_predicted_mean_price=0
def predict_and_compare_func(history):
    global before_predicted_mean_price,FIRST
    X, X_trend, X_seasonal = convert_seasonal_trend(history)
    mean_output = model.predict({"raw": np.array([X]), "trend": np.array([X_trend]), "seasonal": np.array([X_seasonal])})

    mean=0
    for num in mean_output.tolist()[0]:
        print("mean_output:"+str(num))
        mean+=int(num)
    print("before_predicted_mean_price:"+str(before_predicted_mean_price))
    print("mean price:"+str(mean))
    if FIRST == True:
        return_num = 0
        FIRST = False
    else:
        return_num = mean - before_predicted_mean_price

    before_predicted_mean_price=mean

    print("Predicted Value!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(return_num)
    return int(return_num)

def get_price():
    url = 'https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=60'
    res = requests.get(url).json()['result']['60']
    pulled_price=[data_4_thick[1] for data_4_thick in res[0:99]]
    pulled_price.reverse()
    return pulled_price

def bid_ask_function(prediction):
    print(prediction)
    jpy, btc = get_left_money()

    if prediction >= 0.0:
        public_zaif = ZaifPublicApi()
        bids_top_price = public_zaif.depth('btc_jpy')['bids'][0][0]
        order_price = bids_top_price + 5
        bid_ask = "bid"
        plus_minus = 1
        amount = float(abs(round((prediction * jpy * DIVISION_NUM) / order_price, 4)))
        print("Amount!!!!!!!!!!!!!!!")
        print(amount)
    else:
        public_zaif = ZaifPublicApi()
        asks_top_price = public_zaif.depth('btc_jpy')['asks'][0][0]
        order_price = asks_top_price - 5
        bid_ask = "ask"
        plus_minus = -1
        amount = float(abs(round((prediction * btc * DIVISION_NUM), 4)))
        print("prediction order Amount"+str(amount))

    return order_price, plus_minus, bid_ask, amount


def get_left_money():
    global trade_zaif
    temp_info=trade_zaif.get_info()
    jpy=temp_info["funds"]["jpy"]
    btc=temp_info["funds"]["btc"]
    return jpy, btc

#この関数はレバレッジ取引では使わない。
def order(order_price,plus_minus,bid_ask,amount_num):
    #5円は100000分の一の損 0.000001倍　0.0001％１０回位ならトライしても良い。
    global trade_zaif
    for i in range(0,2):
        #もしアクティブなオーダが残ってるようだったら再オーダ
        try:
            order_price+=10*plus_minus*i
            #jpy, btc = get_left_money()
            #leverage_zaif.trade(currency_pair="btc_jpy", action=bid_ask, price=int(order_price),amount=amount_num)
            trade_zaif.create_position(type="futures", leverage=1.5, group_id=1, currency_pair="btc_jpy", action=bid_ask,
                                       price=order_price, amount=amount_num, limit=int(order_price - 10000))
        except:
            print("order Sell_Buy error!!(詳細は以下)")
            print(sys.exc_info()[0])
            time.sleep(10)
            import traceback
            traceback.print_exc()
            continue

        try:
            time.sleep(15)#重要！
            active_pos = trade_zaif.active_positions(type="futures",group_id=1,currency_pair="btc_jpy")
            if list(active_pos.keys()) == []:
                return True
            int_active_pos = int(list(active_pos.keys())[0])
            leverage_zaif.cancel_position(type="futures", group_id=1, leverage_id=int(int_active_pos))
            time.sleep(7)
        except:
            print("order Cancel error!!（詳細は以下）")
            print(sys.exc_info()[0])
            time.sleep(7)
            traceback.print_exc()
            continue



def leverage_order(order_price,plus_minus,bid_ask,amount_num):
    global leverage_zaif
    print("leverage_order LOT total num:"+str(int(amount_num/ ONE_ORDER_AMOUNT_NUM)))
    def order_function(order_price,plus_minus,bid_ask):
        global leverage_zaif
        for idx in range(1, 3):
            try:
                order_price += 10*plus_minus*idx
                #jpy, btc = get_left_money()
                order_log=leverage_zaif.create_position(type="futures", leverage=1, group_id=1, currency_pair="btc_jpy",
                   action=bid_ask, price=int(order_price), amount=ONE_ORDER_AMOUNT_NUM)
                order_id_log=order_log["leverage_id"]
            except:
                print("order Sell_Buy error!!(詳細は以下)")
                print(sys.exc_info()[0])
                traceback.print_exc()
                time.sleep(10)
                continue
            #ここのtry-exceptをわざわざ分けなくて良い
            try:
                time.sleep(15)#重要！
                active_pos = leverage_zaif.active_positions(type="futures", group_id=1, currency_pair="btc_jpy")
                print(active_pos)
                list_active_pos = list(active_pos.keys())
                if order_id_log not in list_active_pos:
                    return True

                leverage_zaif.cancel_position(type="futures", group_id=1, leverage_id=int(order_id_log))
                time.sleep(7)
            except:
                print("order Cancel error!!（詳細は以下）")
                print(sys.exc_info()[0])
                time.sleep(7)
                traceback.print_exc()
                continue

    for idx_out in range(0, int(amount_num / ONE_ORDER_AMOUNT_NUM)):
        print("leverage_order LOT current num:"+str(idx_out))
        print("leverage lot num:"+str(int(amount_num / ONE_ORDER_AMOUNT_NUM)))
        print("amount_num:"+str(amount_num))
        order_function(order_price,plus_minus,bid_ask)

def search_pos_id(pos):
    order_done_flag=False
    pos_ids=[]
    for idx in range(0,len(pos)):
        try:
            pos_id = int(list(pos.keys())[idx])
            pos_ids.append(pos_id)
            done_amount=pos[str(pos_id)]["amount_done"]
            order_done_flag=True
        except:
            traceback.print_exc()
            print(sys.exc_info()[0])
    return pos_ids, order_done_flag


def leverage_complete_order(plus_minus, bid_ask, amount_num):
    #レバレッジ取引の注文を完了(決済)する関数

    try:
        pos = leverage_zaif.active_positions(type="futures", group_id=1, currency_pair="btc_jpy")
        print(pos)
    except:
        traceback.print_exc()
        print(sys.exc_info()[0])

    #done_amountは、信用取引をする際の最初の指値が刺さったかどうかを判断する指標。決済ではない。
    pos_ids, order_done_flag = search_pos_id(pos)

    if order_done_flag is False:#注文して最初の指値が刺さっていなかったら、決済できないのでreturnする
        return

    #できれば複数の建玉を一度に決済したい
    print("Clean Order Lot num:"+str(int(amount_num / ONE_ORDER_AMOUNT_NUM)))
    counter=1
    for id in pos_ids:
        if counter >= int(amount_num / ONE_ORDER_AMOUNT_NUM):
            return
        try:
            counter+=1
            price = pos[str(id)]["price"]
            if bid_ask == "bid":
                print("cleaning order Bid  Try"+str(counter)+"回目")
                bids_top_price = public_zaif.depth('btc_jpy')['bids'][0][0]
                limit_price = bids_top_price + counter*5
            else:
                print("cleaning order Ask  Try"+str(counter)+"回目")
                asks_top_price = public_zaif.depth('btc_jpy')['asks'][0][0]
                limit_price = asks_top_price - counter*5
            leverage_zaif.change_position(type="futures", leverage_id=id, group_id=1, price=int(price), limit=int(limit_price))
            print("Done. cleaning order")
            time.sleep(10)
        except:
            traceback.print_exc()
            print(sys.exc_info()[0])

trade=TradeClass()
history=get_price()
print("History data is here. execute get_price() function")
print(history)

#買いは+、売りは−
leverage_pos_num = -33

for num in range(99999999999):
    #time.sleep(60*5)#本当は5分×60セット待たなければならない
    try:
        time.sleep(20)
        public_zaif = ZaifPublicApi()
        a = public_zaif.last_price(('btc_jpy'))
        print(str(a))
        last_price = int(a['last_price'])
        history.append(last_price)
    except:
        #1回はappend list errorが出る。その時にdef __init__():されてしまって、エラーが２回めでなくなってしまう。
        print("List append error in main loop:", sys.exc_info()[0])
        traceback.print_exc()

    if len(history) <= INPUT_LEN:
        #wait 1 min
        continue
    try:
        del history[0]
        prediction =predict_and_compare_func(np.array(history))#TODO:データの整形
        order_price, plus_minus, bid_ask, amount_num = bid_ask_function(prediction)
        if bid_ask == "bid":
            leverage_pos_num +=1
        else:
            leverage_pos_num -=1

        if amount_num == 0:
            #wait 1 min
            continue
        print("bid_ask:"+str(bid_ask)+" leverage_pos_num:"+str(leverage_pos_num))
        if leverage_pos_num == 0 or (leverage_pos_num >= 1 and bid_ask == "bid") or \
                         (leverage_pos_num <= -1 and bid_ask == "ask"):
            leverage_order(order_price, plus_minus, bid_ask, amount_num)
        else:
            print("Cleaning Order（注文の決済中）")
            leverage_complete_order(plus_minus, bid_ask, amount_num)
            print("Cleaning Order")
    except:
        print("Prediction and Order Area Error in main loop!!", sys.exc_info()[0])
        traceback.print_exc()
        print(sys.exc_info()[0])