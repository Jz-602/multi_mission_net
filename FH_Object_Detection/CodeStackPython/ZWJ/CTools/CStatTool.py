import numpy as np
import matplotlib.pyplot as plt

class StatTool:
    def add_subplot(fig, subplot_ind, hist_input, subplot_name):
        pic = fig.add_subplot(subplot_ind)
        n, bins, patches = plt.hist(hist_input, bins=10)
        for i in range(len(n)):
            plt.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.01, int(n[i]), ha='center', va='bottom')
            # plt.text(bins[i] + (bins[1] - bins[0]) / 2 , n[i] * 1.15, round(n[i] / np.sum(n),3), ha='center', va='bottom',color='r')
            plt.text(bins[i] + (bins[1] - bins[0]) / 2, np.max(n) * 1.1, round(n[i] / np.sum(n), 2), ha='center',
                     va='bottom',
                     color='r')
        plt.title(subplot_name)
        plt.xlabel("probability")
        plt.ylabel("rate")
