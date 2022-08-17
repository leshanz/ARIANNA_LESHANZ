import matplotlib.pyplot as plt
import numpy as np
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
import datetime
from NuRadioReco.utilities import units
import ariannaHelper
import NuRadioReco.detector.detector as detector
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from matplotlib import colors
from scipy.optimize import curve_fit


det = detector.Detector()
det.update(datetime.datetime(2018, 12, 15))
nurFile = []

ratio_list = []
zenith_list = []
weight_list = []
direct = []
refracted = []
reflected = []
pulse1_amp_list = []
pulse2_amp_list = []
data = NuRadioRecoio.NuRadioRecoio(nurFile)
for evt in data.get_events():
    stn = evt.get_station(61)
    sim_stn = stn.get_sim_station()

    zenith, azimuth = ariannaHelper.get_arrival_direction_simulation(stn)
    weight = ariannaHelper.get_weight_simulation(evt)

    chn4_sim_trace1 = np.zeros(256)
    chn4_sim_trace2 = np.zeros(256)

    for chn in sim_stn.get_channels_by_channel_id(4):
        if chn.get_ray_tracing_solution_id() == 0:
            chn4_sim_trace1 += chn.get_trace()
        elif chn.get_ray_tracing_solution_id() == 1:
            chn4_sim_trace2 += chn.get_trace()

    if (not np.isfinite(chn4_sim_trace1.all())) or (not np.isfinite(chn4_sim_trace2.all())):
        continue

    chn4_direct_amplitude = np.max(np.abs(chn4_sim_trace1))/units.mV
    chn4_reflected_amplitude = np.max(np.abs(chn4_sim_trace2)/units.mV)

    if (chn4_reflected_amplitude == 0) or (not np.isfinite(chn4_direct_amplitude)):
        continue
    chn4_ratio = chn4_direct_amplitude/chn4_reflected_amplitude
    ratio = chn4_ratio

    pulse1_amp = chn4_direct_amplitude
    pulse2_amp = chn4_reflected_amplitude

    if ratio == 0 or (not np.isfinite(ratio)):
        continue

    pulse1_amp_list.append(pulse1_amp)
    pulse2_amp_list.append(pulse2_amp)
    ratio_list.append(ratio)
    zenith_list.append(zenith/units.deg)
    weight_list.append(weight)

reflected = np.array(reflected)
refracted = np.array(refracted)
direct = np.array(direct)
zenith_list = np.array(zenith_list)
ratio_list = np.array(ratio_list)
weight_list = np.array(weight_list)
pulse1_amp_list = np.array(pulse1_amp_list)
pulse2_amp_list = np.array(pulse2_amp_list)

mask = zenith_list < 120

binned_ratio_list = {}
binned_mean_list = []
binned_std_list = []
binned_pulse1_amp_list = {}
binned_pulse2_amp_list = {}
binned_pulse1_mean_list = []
binned_pulse1_std_list = []
binned_pulse2_mean_list = []
binned_pulse2_std_list = []

xbins = np.linspace(80, 180, 200)
for c in range(len(xbins)-1):
    binned_ratio_list[c] = []
    binned_pulse1_amp_list[c] = []
    binned_pulse2_amp_list[c] = []
    for i, zenith in enumerate(zenith_list):
        if (zenith > xbins[c]) and (zenith < xbins[c+1]):
            binned_ratio_list[c].append(ratio_list[i])
            binned_pulse1_amp_list[c].append(pulse1_amp_list[i])
            binned_pulse2_amp_list[c].append(pulse2_amp_list[i])

    binned_mean_list.append(np.mean(binned_ratio_list[c]))
    binned_std_list.append(np.std(binned_ratio_list[c]))
    binned_pulse1_mean_list.append(np.mean(binned_pulse1_amp_list[c]))
    binned_pulse1_std_list.append(np.std(binned_pulse1_amp_list[c]))
    binned_pulse2_mean_list.append(np.mean(binned_pulse2_amp_list[c]))
    binned_pulse2_std_list.append(np.std(binned_pulse2_amp_list[c]))

binned_mean_list = np.array(binned_mean_list)

zenith_mask = (xbins[:-1] > 125) * (xbins[:-1] < 143)


def exponential(x, A, b, c):
    return A * np.exp(b*(x-c)) + 1


def power_law(x, A, b, n):
    return A*(x-b)**n + 1


[A, b, c], _ = curve_fit(lambda x, A, b, c: A *
                         np.exp(b*(x-c)), xbins[:-1][zenith_mask], binned_mean_list[zenith_mask], p0=[1, 1, 130])


fig, ax = plt.subplots()
ax.scatter(xbins[:-1], binned_mean_list, s=3, label=r'$V_1/V_2$')
ax.plot(xbins[:-1][zenith_mask], exponential(xbins[:-1]
        [zenith_mask], A, b, c), lw=2, label='Ae^(b(x-c))', c='red', ls='--')
ax.set_xlabel('zenith [deg]')
ax.set_ylabel(r'$V_1/V_2$')
# ax.set_yscale('log')
ax.set_ylim(0.5, 1.5)
ax.set_title(f'A={A:.2f} b={b:.2f} c={c:.2f}')
ax.legend()
# fig.savefig('plots/simulation/stn61_sim_zenith_dipole_pulse_amp_ratio.png')
plt.show()
