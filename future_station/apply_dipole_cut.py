import numpy as np
import os
import re
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import ariannaHelper
from NuRadioReco.utilities.geometryUtilities import get_fresnel_r_p, get_fresnel_r_s
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.detector.detector as detector
import datetime
import NuRadioReco.modules.channelResampler
from radiotools.helper import get_normalized_xcorr
from matplotlib import colors
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.utilities.io_utilities import read_pickle
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
from NuRadioReco.modules import channelTimeWindow as cTWindow
import NuRadioReco.modules.channelSignalReconstructor

cTW = cTWindow.channelTimeWindow()
cTW.begin(debug=False)


det = detector.Detector()
det.update(datetime.datetime(2019, 1, 1))
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()

allowrance = 10


def calculate_time_window(zenith):
    return 1.2340284077008037 * zenith - 103.47292624311076 - allowrance, 1.2340284077008037 * zenith - 103.47292624311076 + allowrance


def get_pulse_ratio(zenith):
    if zenith < 130:
        return 1
    elif zenith < 143:
        return 1.0/(0.19 * np.exp(0.64 * (zenith - 133.49))+1)
    else:
        return 0


def get_deltaT(zenith, azimuth):
    A = 18.544278656417784
    B = 27.175991087647716
    k = 1.0084359501208893
    a = 2.3927807926085283
    b = 1.5079140357258132
    c = 6.165506618688502
    d = -11.88895669544745
    return A*np.sin(k*np.deg2rad(azimuth)+a) + B*np.sin(b*np.deg2rad(zenith) + c)+d - allowrance, A*np.sin(k*np.deg2rad(azimuth)+a) + B*np.sin(b*np.deg2rad(zenith) + c)+d + allowrance


def get_signal_start_time(trace):

    xcorr_trace = get_normalized_xcorr(trace, lpda_template)
    xcorr_pos = np.argmax(np.abs(xcorr_trace))

    tplt_pos = np.argmax(lpda_template)

    trace = np.max(np.abs(lpda_template))*trace/np.max(np.abs(trace))
    signal_time = 1 + xcorr_pos - len(lpda_template) + tplt_pos

    return signal_time/10


def dipole_cut(x):
    kargs = [0.00025045, -0.01117146, 0.16539936, -0.11540729]
    y = 0
    for c, i in enumerate(kargs):
        y += i * (x ** (len(kargs)-1-c))

    if y > 0.7:
        y = 0.7

    return y


template_trace = np.load(
    '', allow_pickle=True)


lpda_template = np.load(
    '', allow_pickle=True)
tplt_max_index = np.argmax(np.abs(template_trace))
simulation_files = ['']
sim = NuRadioRecoio.NuRadioRecoio(simulation_files)

xcorr_list_sim = []
zenith_list_sim = []
weight_list = []
dipole_amplitude_list_sim = []


for evt in sim.get_events():
    stn = evt.get_station(61)
    run_num = evt.get_run_number()
    evt_id = evt.get_id()

    true_zenith, true_azimuth = ariannaHelper.get_arrival_direction_simulation(
        stn)
    zenith = stn[stnp.zenith]
    azimuth = stn[stnp.azimuth]
    channelResampler.run(evt, stn, det, 10*units.GHz)

    chrenkov_angle = ariannaHelper.get_chrenkov_angle_simulation(stn)
    weight = ariannaHelper.get_weight_simulation(evt)

    time_window_low, time_window_high = calculate_time_window(zenith/units.deg)

    tplt_window_low, tplt_window_high = get_deltaT(
        zenith/units.deg, azimuth/units.deg)

    trigger = stn.get_trigger('2of4_3.94sigma')
    triggered_chns = trigger.get_triggered_channels()

    chn = stn.get_channel(triggered_chns[0])
    times = chn.get_times()/units.ns
    trigger_time = trigger.get_trigger_time() - times[0]
    first_signal_time_lpda = trigger_time

    temp_list = []
    for i in triggered_chns:
        chn = stn.get_channel(i)
        trace = chn.get_trace()
        temp_list.append(get_signal_start_time(trace))
    first_signal_time_lpda = np.min(temp_list)
    used_channel = triggered_chns[np.argmin(temp_list)]
    used_channel_times = stn.get_channel(used_channel).get_times()/units.ns
    if (time_window_high < 0) or (time_window_low < 0):
        time_window_low = 0
        time_window_high = allowrance*2

    pulse_ratio = get_pulse_ratio(zenith/units.deg)

    weight_list.append(weight)

    xcorr = 0
    xcorr_pos = 0
    tplt = 0
    chn = stn.get_channel(4)
    trace = chn.get_trace()
    dipole_amp = np.max(np.abs(trace))/units.mV
    dipole_amplitude_list_sim.append(dipole_amp)

    chn4_times = chn.get_times()/units.ns

    for i in np.linspace(time_window_low, time_window_high, 100):

        template_trace_copy = np.append(
            np.zeros(int(i*10)), template_trace)
        template_trace_copy *= pulse_ratio
        template_trace_copy2 = np.resize(
            template_trace, template_trace_copy.shape)
        template = template_trace_copy2 + template_trace_copy

        tplt_pos_low = int(-1 + 10*(first_signal_time_lpda +
                                    tplt_window_low)-tplt_max_index + len(template))
        tplt_pos_high = int(-1 + 10*(first_signal_time_lpda +
                            tplt_window_high)-tplt_max_index + len(template))

        xcorr_temp = np.max(np.abs(get_normalized_xcorr(
            trace, template))[tplt_pos_low:tplt_pos_high])

        xcorr_pos_temp = np.argmax(np.abs(get_normalized_xcorr(
            trace, template))[tplt_pos_low:tplt_pos_high])+tplt_pos_low

        if xcorr_temp > xcorr:
            xcorr = xcorr_temp
            xcorr_pos = xcorr_pos_temp
            tplt = template
    zenith_list_sim.append(zenith/units.deg)

    xcorr_list_sim.append(xcorr)
np.save('',
        [dipole_amplitude_list_sim, xcorr_list_sim, weight_list])

dipole_amplitude_list_sim = np.array(dipole_amplitude_list_sim)
xcorr_list_sim = np.array(xcorr_list_sim)
weight_list = np.array(weight_list)


files = ['']
data = NuRadioRecoio.NuRadioRecoio(files)

xcorr_list_data = []
zenith_list_data = []
dipole_amplitude_list_data = []
updown_cut_list = []
dipole_cut_list = []
lpda_max_amp_list = []
lpda_xcorr_list = []

for evt in data.get_events():
    stn = evt.get_station(61)
    run_num = evt.get_run_number()
    evt_id = evt.get_id()

    channelResampler.run(evt, stn, det, 10*units.GHz)

    zenith = stn[stnp.zenith]
    azimuth = stn[stnp.azimuth]
    time_window_low, time_window_high = calculate_time_window(zenith/units.deg)

    tplt_window_low, tplt_window_high = get_deltaT(
        zenith/units.deg, azimuth/units.deg)

    trigger = stn.get_trigger('2of4_4.3sigma')
    triggered_chns = trigger.get_triggered_channels()

    temp_list = []
    for i in triggered_chns:
        chn = stn.get_channel(i)
        trace = chn.get_trace()
        temp_list.append(get_signal_start_time(trace))
    first_signal_time_lpda = np.min(temp_list)

    if (time_window_high < 0) or (time_window_low < 0):
        time_window_low = 0
        time_window_high = allowrance*2

    pulse_ratio = get_pulse_ratio(zenith/units.deg)

    chn = stn.get_channel(4)
    trace = chn.get_trace()
    xcorr = 0
    dipole_amp = np.max(np.abs(trace))/units.mV
    dipole_amplitude_list_data.append(
        dipole_amp)

    for i in np.linspace(time_window_low, time_window_high, 100):
        template_trace_copy = np.append(
            np.zeros(int(i*10)), template_trace)
        template_trace_copy *= pulse_ratio
        template_trace_copy2 = np.resize(
            template_trace, template_trace_copy.shape)
        template = template_trace_copy2 + template_trace_copy

        tplt_pos_low = int(-1 + 10*(first_signal_time_lpda +
                                    tplt_window_low)-tplt_max_index + len(template))
        tplt_pos_high = int(-1 + 10*(first_signal_time_lpda +
                            tplt_window_high)-tplt_max_index + len(template))

        xcorr_temp = np.max(np.abs(get_normalized_xcorr(
            trace, template)[tplt_pos_low:tplt_pos_high]))

        xcorr_pos_temp = np.argmax(np.abs(get_normalized_xcorr(
            trace, template))[tplt_pos_low:tplt_pos_high])+tplt_pos_low

        if xcorr_temp > xcorr:
            xcorr = xcorr_temp
            xcorr_pos = xcorr_pos_temp
            tplt = template
    zenith_list_data.append(zenith/units.deg)

    xcorr_list_data.append(xcorr)

    updown_cut_list.append(ariannaHelper.apply_updown_cut(stn))

    dipole_cut_list.append(xcorr > dipole_cut(dipole_amp/10.5))

    lpda_max_amp_list.append(
        ariannaHelper.get_max_downward_amplitude(stn)/units.mV)
    lpda_xcorr = stn[stnp.nu_xcorrelations]['nu_avg_xcorr_parallel_nuchannels']
    lpda_xcorr_list.append(lpda_xcorr)

np.save('results/stn61_data_lpda_amp_chi_updown_dipole_cut',
        [lpda_max_amp_list, lpda_xcorr_list, updown_cut_list, dipole_cut_list])


updown_cut_list = np.array(updown_cut_list)
xcorr_list_data = np.array(xcorr_list_data)
dipole_amplitude_list_data = np.array(dipole_amplitude_list_data)


mean = np.average(xcorr_list_sim, weights=weight_list)

mean_data = np.mean(xcorr_list_data)

dipole_cutline = np.load(
    '', allow_pickle=True)
SNRs = dipole_cutline[0]
cutline = dipole_cutline[1]

temp_list = []
temp_list2 = []
for i in range(len(SNRs)-1):
    for c, j in enumerate(dipole_amplitude_list_sim):
        if SNRs[i] < j/10.5 < SNRs[i+1]:
            if xcorr_list_sim[c] > cutline[i]:
                temp_list.append(weight_list[c])
            temp_list2.append(weight_list[c])

yspace = np.linspace(0, 1, 50)
xspace = np.logspace(np.log10(1.0), np.log10(100), 50)
fig, ax = plt.subplots()
ax.set_title(
    'stn61 sim & data dipole correlation')
h = ax.hist2d(dipole_amplitude_list_sim/10.5, xcorr_list_sim, bins=(
    xspace, yspace), cmap='Blues', weights=weight_list)
# ax.scatter(dipole_amplitude_list_data[updown_cut_list]/10.5, xcorr_list_data[updown_cut_list], s=5,
#           label=f'data: #evts={len(dipole_amplitude_list_data[updown_cut_list])}', color='red')
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('simulated event density')
# h2 = ax.hist2d(dipole_amplitude_list_data[updown_cut_list]/10.5, xcorr_list_data[updown_cut_list], bins=(
#    xspace, yspace), cmap='Reds', norm=colors.LogNorm())
#cbar2 = fig.colorbar(h2[3], ax=ax)
#cbar2.set_label('data event density')
ax.plot(SNRs, cutline, lw=3,
        color='b', label='94.8% dipole cut')
#ax.set_xlim(0, 500)
ax.set_ylim(0, 1)
ax.set_xscale('log')
ax.set_xlabel('dipole SNR')
ax.set_ylabel(r'dipole $\chi$')
ax.legend()
# fig.savefig(
#    'plots/stn61_data_sim_dipole_amplitude_dipole_xcorr_tplt_restricted_reco_dir_updown.png')

plt.show()
