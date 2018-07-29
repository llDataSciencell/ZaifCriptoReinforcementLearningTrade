from agent.agent import Agent
from functions import *
import sys

print("aaaa")

window_size, episode_count = int(10), int(1000)
print(stock_name)
print(window_size)
print(episode_count)

agent = Agent(window_size)
data = read_bitflyer_json()  # getStockDataVec(stock_name)

length_data = len(data) - 1
batch_size = window_size

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getStateFromCsvData(data, 0, window_size)
    print(state)
    total_profit = 0
    agent.inventory = []

    for idx in range(length_data):
        action = agent.act(state)

        #TODO idx + 1出なくて良いか？　バグの可能性あり。
        next_state = getStateFromCsvData(data, idx+1, window_size)
        reward = 0
        if action == 1:  # buy
            agent.inventory.append(data[idx])
            print("Buy: " + formatPrice(data[idx]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[idx] - bought_price, 0)
            total_profit += data[idx] - bought_price
            print("Sell: " + formatPrice(data[idx]) + " | Profit: " + formatPrice(data[idx] - bought_price))

        done = True if idx == length_data - 1 else False

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if idx % 100:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    agent.model.save("models/model_ep" + str(e))

