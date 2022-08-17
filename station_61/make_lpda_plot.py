from aenum import enum
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import ariannaHelper
from scipy.optimize import curve_fit
import random
from scipy.signal import savgol_filter
from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.utilities.units as units


def dipole_cut(x):
    SNRs = dipole_cutline[0]
    cutline = dipole_cutline[1]

    for i in range(len(SNRs)-1):
        if ((x > SNRs[i]) and (x < SNRs[i+1])):
            return cutline[i]
    return cutline[-1]


def func2(x, *kargs):
    y = np.zeros_like(x)
    print(kargs)
    for c, i in enumerate(kargs[0]):
        y += i * (x ** (len(kargs[0])-1-c))

    return y


def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x-mu)/sigma)**2)


simu_max_amp_list = []
simu_xcorr_list = []
weight_list = []

rms_noise_down = 10.5*units.mV

simulation_files = []

simu_data = NuRadioRecoio(simulation_files)
for evt in simu_data.get_events():
    stn = evt.get_station(61)
    xcorr = stn[stnp.nu_xcorrelations]['nu_avg_xcorr_parallel_nuchannels']
    simu_xcorr_list.append(xcorr)
    simu_max_amp_list.append(ariannaHelper.get_max_downward_amplitude(stn))
    weight_list.append(ariannaHelper.get_weight_simulation(evt))

simu_xcorr_list = np.array(simu_xcorr_list)
simu_max_amp_list = np.array(simu_max_amp_list)
simu_SNR_list = simu_max_amp_list / (rms_noise_down)
weight_list = np.array(weight_list)
sim_mask = (simu_SNR_list > 4.0)

data = np.load('',
               allow_pickle=True)
dipole_amplitude_list_data = np.array(data[0])
dipole_snr_list_data = dipole_amplitude_list_data/10.5
xcorr_list_data = data[1]
updown_cut_list = data[2]
updown_cut_list = np.array([bool(i) for i in updown_cut_list])
dipole_cut_list = []

dipole_cutline = np.load(
    '', allow_pickle=True)


for c, i in enumerate(dipole_snr_list_data):
    dipole_cut_list.append(xcorr_list_data[c] > dipole_cut(i))

dipole_cut_list = np.array(dipole_cut_list)

data2 = np.load(
    '', allow_pickle=True)

SNR_list_all = np.array(data2[0])
xcorr_list_all = np.array(data2[1])

mask = dipole_cut_list*updown_cut_list


SNR_bin_list = np.append(np.linspace(4, 10, 3), np.linspace(20, 100, 9))

efficiency_cut = []
for i in SNR_bin_list:

    temp_list = []
    temp_weight_list = []
    for c, j in enumerate(simu_SNR_list):
        if i < j < i+1:
            temp_list.append(simu_xcorr_list[c])
            temp_weight_list.append(weight_list[c])

    if len(temp_list) == 0:
        efficiency_cut.append(0)
        continue

    cut_threshold = ariannaHelper.weighted_quantile(
        temp_list, 0.045, temp_weight_list)
    efficiency_cut.append(cut_threshold)


efficiency_cut = np.array(efficiency_cut)
efficiency_cut[np.where((SNR_bin_list > 4)*(SNR_bin_list < 10))] += 0.03
cutline = savgol_filter(efficiency_cut, 7, 2)
cutline[np.where(SNR_bin_list > 40)] = 0.55

SNR_list = SNR_list_all[mask]
xcorr_list_sim = simu_xcorr_list

binned_xcorr_list = []
binned_sim_xcorr_list = []
binned_sim_weight_list = []
ratio_list = []
randomly_selected_xcorr_list = []
randomly_selected_SNR_list = []
SNR_bin_list = np.linspace(2, 100, 40)
print('SNR_bin_list=', SNR_bin_list)
for count in range(len(SNR_bin_list)-1):
    temp_list = []
    temp_sim_list = []
    temp_sim_weight_list = []

    for c, j in enumerate(SNR_list_all[updown_cut_list]):
        if SNR_bin_list[count] < j < SNR_bin_list[count+1]:
            temp_list.append(xcorr_list_all[updown_cut_list][c])
    for c, j in enumerate(simu_SNR_list):
        if SNR_bin_list[count] < j < SNR_bin_list[count+1]:
            temp_sim_list.append(xcorr_list_sim[c])
            temp_sim_weight_list.append(weight_list[c])
    temp_list = np.array(temp_list)
    temp_sim_list = np.array(temp_sim_list)
    temp_sim_weight_list = np.array(temp_sim_weight_list)
    binned_xcorr_list.append(temp_list)
    binned_sim_xcorr_list.append(temp_sim_list)
    binned_sim_weight_list.append(temp_sim_weight_list)

for k in range(len(SNR_list_all[mask])*1000):
    index = random.randint(0, len(SNR_list_all[updown_cut_list])-1)
    randomly_selected_xcorr_list.append(xcorr_list_all[updown_cut_list][index])
    randomly_selected_SNR_list.append(SNR_list_all[updown_cut_list][index])


fig, ax = plt.subplots()
# fig.suptitle(
#    'stn61 data & simulation up-down & dipole cuts w/ zenith dependent time window')
# ax.set_title(f'#evts={len(xcorr_list_all)}')
plt.grid(False)
ax.set_ylim(0, 1)
ax.set_ylabel(r'$\chi$')

xspace = np.logspace(np.log10(1.0), np.log10(130), 150)
yspace = np.linspace(0, 1, 70)

h = ax.hist2d(simu_SNR_list[sim_mask], simu_xcorr_list[sim_mask], bins=(
    xspace, yspace), cmap='Blues', norm=colors.LogNorm(), cmin=1e-1, weights=weight_list[sim_mask])
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('simulated event density')

ax.scatter(SNR_list_all, xcorr_list_all,
           label=f'all events:{len(SNR_list_all)}', s=2, color='0.3', alpha=0.5)
ax.scatter(SNR_list_all[updown_cut_list], xcorr_list_all[updown_cut_list],
           label=fr'99% updown cut:{len(SNR_list_all[updown_cut_list])}', s=4, color='0.3', alpha=0.7, marker='^')
# ax.scatter(randomly_selected_SNR_list, randomly_selected_xcorr_list,
#           label=f'randomly selected:{len(randomly_selected_SNR_list)}', s=6, color='green', alpha=0.5)
h2 = ax.hist2d(randomly_selected_SNR_list, randomly_selected_xcorr_list, bins=(
    xspace, yspace), cmin=1, norm=colors.LogNorm())
cbar = fig.colorbar(h2[3], ax=ax)
cbar.set_label('randomly selected event density')
ax.scatter(SNR_list_all[mask], xcorr_list_all[mask], s=9,
           color='red', label=f'94.8% dipole cut:{len(SNR_list_all[mask])}')
SNR_bin_list = np.append(np.linspace(4, 10, 3), np.linspace(20, 100, 9))

temp_list = []
temp_list2 = []
for i in range(len(SNR_bin_list)-1):
    for c, j in enumerate(simu_SNR_list[sim_mask]):
        if SNR_bin_list[i] < j < SNR_bin_list[i+1]:
            if simu_xcorr_list[sim_mask][c] > cutline[i]:
                temp_list.append(weight_list[sim_mask][c])
            temp_list2.append(weight_list[sim_mask][c])

ax.plot(SNR_bin_list, cutline,
        lw=3, ls='--', color='black', label=r'97.1% neutrino efficiency')
ax.legend(loc=1)
ax.set_xlabel(r'SNR')
ax.set_xlim(3, 150)
ax.set_ylim(0, 1)
ax.set_xscale('log')
# fig.savefig(
#    'plots/stn61_data_sim_updown_dipole_cut.png')
plt.show()
