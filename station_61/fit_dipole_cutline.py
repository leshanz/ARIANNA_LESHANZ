from scipy.optimize import curve_fit
import ariannaHelper
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def poly(x, *kargs):
    y = np.zeros_like(x)
    print(kargs)
    for c, i in enumerate(kargs[0]):
        y += i * (x ** (len(kargs[0])-1-c))

    return y


cut_file = np.load(
    '', allow_pickle=True)
SNR_bin_list = np.array(cut_file[0])
cutline = np.array(cut_file[1])
cutline[np.where(SNR_bin_list > 100)] = 0.7


fitline = savgol_filter(cutline, 7, 2)
np.save('', [SNR_bin_list, fitline])
plt.plot(SNR_bin_list, cutline)
plt.plot(SNR_bin_list, fitline)
plt.show()
