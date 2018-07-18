# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

def main():
    date, data = np.array([0,1,2,3,4,5]), np.array([1,2,3,4,5,6])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(date, data)
    ax.set_title("BTC Price")
    ax.set_xlabel("Day")
    ax.set_ylabel("BTC Price[$]")
    plt.grid(fig)
    plt.show(fig)

main()
