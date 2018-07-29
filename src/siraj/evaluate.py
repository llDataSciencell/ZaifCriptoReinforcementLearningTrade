import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 2:
	print("Usage: python evaluate.py [model]")
	exit()

model_name = sys.argv[1]
model = load_model("models/" + model_name)
window_size = int(10)
print(model.layers[0].input.shape.as_list())

agent = Agent(window_size, True, model_name)
data = read_bitflyer_json()
l = len(data) - 1
batch_size = 32

state = getStateFromCsvData(data, 0, window_size)
total_profit = 0
agent.inventory = []

for t in range(10000,l):
	action = agent.act(state)

	# sit
	next_state = getStateFromCsvData(data, t + 1, window_size)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if t % 100 == 0:
		print("--------------------------------")
		print(" Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")