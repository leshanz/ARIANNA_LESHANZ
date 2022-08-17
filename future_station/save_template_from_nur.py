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
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import pickle
import NuRadioReco.modules.triggerTimeAdjuster

det = detector.Detector()
det.update(datetime.datetime(2019, 1, 1))
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin()

files = []

data = NuRadioRecoio.NuRadioRecoio(files)
template_trace_dict = {}

zen_ref = np.deg2rad(120)  # stn61 & 51
az_ref = np.deg2rad(27.5)  # stn 61
c_ref = np.deg2rad(1.)


for evt in data.get_events():
    stn = evt.get_station(61)
    sim_stn = stn.get_sim_station()

    event_id = evt.get_id()
    run_number = evt.get_run_number()
    energy = ariannaHelper.get_particle_energy_simulation(evt)
    zenith, azimuth = ariannaHelper.get_arrival_direction_simulation(
        stn)
    distance = ariannaHelper.get_vertex_radius_simulation(evt)/units.km
    chrenkov_angle = ariannaHelper.get_chrenkov_angle_simulation(
        stn)
    weight = ariannaHelper.get_weight_simulation(evt)
    r_p = get_fresnel_r_p(np.pi-zenith, n_1=1.3, n_2=1.)
    r_s = get_fresnel_r_s(np.pi-zenith, n_1=1.3, n_2=1.)
    if not ((zenith > 113*units.deg) and (zenith < 117*units.deg) and (chrenkov_angle > 57*units.deg) and (chrenkov_angle < 58*units.deg)):
        continue
    if not ((azimuth > 20*units.deg) and (azimuth < 40*units.deg)):
        continue
    if not (weight > 0.5):
        continue

    triggerTimeAdjuster.run(evt, stn, det)
    triggerTimeAdjuster.run(evt, sim_stn, det)

    channelBandPassFilter.run(
        evt, stn, det, [80*units.MHz, 500*units.MHz], 'butter', 10)
    channelBandPassFilter.run(
        evt, sim_stn, det, [80*units.MHz, 500*units.MHz], 'butter', 10)

    fig = plt.figure(figsize=(7, 5))
    axes = fig.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
    fig.suptitle(
        f'Station 61 Run {run_number} Event {event_id} weight={weight:.3f} \n energy=10^{np.log10(energy):.2f}eV zenith={zenith/units.deg:.2f} azimuth={azimuth/units.deg:.2f}\n vertex distance={distance:.2f} viewing angle={57-chrenkov_angle/units.deg:.2f} r_p={r_p:.2f} r_s={r_s:.2f}')

    for i in range(8):
        chn = stn.get_channel(i)
        if i == 0:
            ax = axes[0, 0]
            color = 'red'
            antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 1:
            ax = axes[0, 1]
            color = 'green'
            antenna_type = 'downward LPDA'
        elif i == 2:
            ax = axes[1, 0]
            color = 'blue'
            antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 3:
            ax = axes[1, 1]
            color = 'C1'
            antenna_type = 'downward LPDA'
        elif i == 4:
            ax = axes[2, 0]
            color = '0.5'
            antenna_type = 'dipole'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 5:
            ax = axes[2, 1]
            color = '0.5'
            antenna_type = 'upward LPDA'
        elif i == 6:
            ax = axes[3, 0]
            color = '0.5'
            antenna_type = 'dipole'
            ax.set_ylabel('Amplitude (mV)')
            ax.set_xlabel('Time (ns)')
        elif i == 7:
            ax = axes[3, 1]
            color = '0.5'
            antenna_type = 'upward LPDA'
            ax.set_xlabel('Time (ns)')
        ax.set_title(r'channel{} {}'.format(
            i, antenna_type))
        time = chn.get_times()/units.ns
        amplitude = chn.get_trace()/units.mV
        max_index = np.argmax(np.abs(amplitude))
        ax.plot(time, amplitude, color=color, lw=1)
    # fig.savefig(
    #    f'../future_station/plots/future_station_sim_tplt.png')
    plt.show()
    template_trace_dict[zen_ref] = {}
    template_trace_dict[zen_ref][az_ref] = {}
    template_trace_dict[zen_ref][az_ref][c_ref] = {}
    for i in range(4):
        chn = stn.get_channel(1)
        template_trace_dict[zen_ref][az_ref][c_ref][i] = chn.get_trace()
    for i in [4, 6]:
        chn = stn.get_channel(4)
        template_trace_dict[zen_ref][az_ref][c_ref][i] = chn.get_trace()
    for i in [5, 7]:
        chn = stn.get_channel(7)
        template_trace_dict[zen_ref][az_ref][c_ref][i] = chn.get_trace()

    with open('', 'wb') as handle:
        pickle.dump(template_trace_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    channelResampler.run(evt, stn, det, 10*units.GHz)
    channelResampler.run(evt, sim_stn, det, 10*units.GHz)

    dipole_tplt = np.zeros(2560)
    for chn in sim_stn.get_channels_by_ray_tracing_id(0):
        if chn.get_id() == 4:
            dipole_tplt += chn.get_trace()
    lpda_tplt = stn.get_channel(0).get_trace()

    plt.plot(range(len(dipole_tplt)), dipole_tplt)
    plt.show()
    break
np.save('', lpda_tplt)
np.save('', dipole_tplt)
