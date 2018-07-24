import numpy as np
import math
import os
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

# returns an an n-day state representation ending at time t
#入力データを作る際も、価格ごとの差分をとってシグモイドに入れている。
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(int(block[i + 1] - block[i])*0.01))

	return np.array([res])
