import numpy as np
import NuRadioReco.utilities.units as units
import NuRadioReco.framework.event as event
import NuRadioReco.framework.station as station
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import particleParameters as pp
from radiotools import helper as hp
import NuRadioReco.detector.detector as detector
import datetime
import matplotlib.pyplot as plt

det = detector.Detector()


def get_max_downward_channel(stn: station.Station):

    max_amp_list = []

    station_id = stn.get_id()
    if station_id == 61:
        downward_channel_ids = [0, 1, 2, 3]
    else:
        raise ValueError('stations other than 61 has not been implemented')

    for i in downward_channel_ids:
        chn = stn.get_channel(i)
        trace = chn.get_trace()
        max_amp_list.append(np.max(np.abs(trace)))
    max_chn_down = downward_channel_ids[np.argmax(max_amp_list)]

    return max_chn_down


def get_max_upward_channel(stn: station.Station):

    max_amp_list = []

    station_id = stn.get_id()
    if station_id == 61:
        upward_channel_ids = [5, 7]
    else:
        raise ValueError('stations other than 61 has not been implemented')

    for i in upward_channel_ids:
        chn = stn.get_channel(i)
        trace = chn.get_trace()
        max_amp_list.append(np.max(np.abs(trace)))
    max_chn_up = upward_channel_ids[np.argmax(max_amp_list)]

    return max_chn_up


def get_max_dipole_channel(stn: station.Station):

    max_amp_list = []

    station_id = stn.get_id()
    if station_id == 61:
        dipole_channel_ids = [4, 6]
    else:
        raise ValueError('stations other than 61 has not been implemented')

    for i in dipole_channel_ids:
        chn = stn.get_channel(i)
        trace = chn.get_trace()
        max_amp_list.append(np.max(np.abs(trace)))
    max_chn_dipole = dipole_channel_ids[np.argmax(max_amp_list)]

    return max_chn_dipole


def get_max_dipole_amplitude(stn: station.Station):
    max_chn_id = get_max_dipole_channel(stn)
    max_chn = stn.get_channel(max_chn_id)
    return np.max(np.abs(max_chn.get_trace()))


def get_max_upward_amplitude(stn: station.Station):
    max_chn_id = get_max_upward_channel(stn)
    max_chn = stn.get_channel(max_chn_id)
    return np.max(np.abs(max_chn.get_trace()))


def get_fluence_channel(stn: station.Station, chn_id: int):

    chn = stn.get_channel(chn_id)
    trace = chn.get_trace()
    fluence = np.sum(trace**2)

    return fluence


def get_max_downward_amplitude(stn: station.Station):

    temp_list = []

    station_id = stn.get_id()
    if station_id == 61:
        downward_channel_ids = [0, 1, 2, 3]
    else:
        raise ValueError('stations other than 61 has not been implemented')

    for i in downward_channel_ids:
        chn = stn.get_channel(i)
        temp_list.append(np.max(np.abs(chn.get_trace())))

    return np.max(temp_list)


def get_arrival_direction_simulation(stn: station.Station):
    sim_stn = stn.get_sim_station()
    e_field = sim_stn.get_electric_fields()[0]
    zenith = e_field[efp.zenith]
    azimuth = e_field[efp.azimuth]

    return zenith, azimuth


def get_vertex_radius_simulation(evt: event.Event):
    shower = evt.get_first_sim_shower()
    distance = ((shower[shp.vertex][0])**2 + (shower[shp.vertex][1])**2)**0.5

    return distance


def get_vertex_distance_simulation(evt: event.Event):
    shower = evt.get_first_sim_shower()
    distance = ((shower[shp.vertex][0])**2 + (shower[shp.vertex]
                [1])**2 + (shower[shp.vertex][2])**2)**0.5

    return distance


def get_chrenkov_angle_simulation(stn: station.Station):
    sim_stn = stn.get_sim_station()
    e_field = sim_stn.get_electric_fields()[0]
    chrenkov_angle = e_field[efp.nu_viewing_angle]

    return chrenkov_angle


def get_particle_energy_simulation(evt: event.Event):
    particle = evt.get_primary()
    energy = particle[pp.energy]

    return energy


def get_weight_simulation(evt: event.Event):
    particle = evt.get_primary()
    weight = particle[pp.weight]

    return weight


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def get_noiseless_trace(stn: station.Station):

    sim_stn = stn.get_sim_station()

    temp_dict = {}
    for i in range(8):
        temp_dict[i] = np.array([0.0] * 256)

    for chn in sim_stn.iter_channels():
        chn_id = chn.get_id()
        temp_dict[chn_id] += chn.get_trace()
    return temp_dict


def get_max_xcorr(trace1, trace2):
    if (np.sum(trace1**2) == 0) or (np.sum(trace2**2) == 0):
        return 0.0
    else:
        xcorr_trace = hp.get_normalized_xcorr(trace1, trace2)

    return np.max(np.abs(xcorr_trace))


def get_amplitude_channel(stn: station.Station, chn_id: int):

    chn = stn.get_channel(chn_id)
    trace = chn.get_trace()
    amplitude = np.max(np.abs(trace))

    return amplitude


def get_max_avg_amplitude_parallel_nu_channels(stn: station.Station):

    station_id = stn.get_id()
    if station_id == 61:
        upward_channel_ids = [5, 7]
    else:
        raise ValueError('stations other than 61 has not been implemented')

    max_down_channel = get_max_downward_channel(stn)
    if max_down_channel in [0, 2]:
        amplitude = np.mean([get_amplitude_channel(
            stn, 0), get_amplitude_channel(stn, 2)])
    elif max_down_channel in [1, 3]:
        amplitude = np.mean([get_amplitude_channel(
            stn, 1), get_amplitude_channel(stn, 3)])

    return amplitude


def apply_updown_cut(stn: station.Station):
    max_upward_amp = get_max_upward_amplitude(stn)/units.mV
    downward_amp = get_max_avg_amplitude_parallel_nu_channels(stn)/units.mV

    return (max_upward_amp < (0.9 * downward_amp + 29.34))


def plot_waveform(stn: station.Station):
    stn_id = stn.get_id()
    n_chns = det.get_number_of_channels(stn_id)
    stn_time = stn.get_station_time()
    # det.update(stn_time)

    if n_chns == 4:
        fig = plt.figure(figsize=(7, 5))
        axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    elif n_chns == 8:
        fig = plt.figure(figsize=(7, 5))
        axes = fig.subplots(nrows=4, ncols=2, sharex=True, sharey=True)

    for i in range(n_chns):
        chn = stn.get_channel(i)
        if i == 0:
            ax = axes[0, 0]
            color = 'red'
            #antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 1:
            ax = axes[0, 1]
            color = 'green'
            #antenna_type = 'downward LPDA'
        elif i == 2:
            ax = axes[1, 0]
            color = 'blue'
            #antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 3:
            ax = axes[1, 1]
            color = 'C1'
            #antenna_type = 'downward LPDA'
        elif i == 4:
            ax = axes[2, 0]
            color = '0.5'
            #antenna_type = 'dipole'
            ax.set_ylabel('Amplitude (mV)')
        elif i == 5:
            ax = axes[2, 1]
            color = '0.5'
            #antenna_type = 'upward LPDA'
        elif i == 6:
            ax = axes[3, 0]
            color = '0.5'
            #antenna_type = 'dipole'
            ax.set_ylabel('Amplitude (mV)')
            ax.set_xlabel('Time (ns)')
        elif i == 7:
            ax = axes[3, 1]
            color = '0.5'
            #antenna_type = 'upward LPDA'
            ax.set_xlabel('Time (ns)')
        ax.set_title(r'channel{}'.format(
            i))
        time = chn.get_times()/units.ns
        amplitude = chn.get_trace()/units.mV
        #amplitude = chn.get_hilbert_envelope()/units.mV
        max_index = np.argmax(amplitude)
        ax.plot(time, amplitude, color=color, lw=1)

    return fig, axes


def plot_frequency_spectrum(stn: station.Station):
    stn_id = stn.get_id()
    n_chns = det.get_number_of_channels(stn_id)
    stn_time = stn.get_station_time()
    # det.update(stn_time)

    if n_chns == 4:
        fig = plt.figure(figsize=(7, 5))
        axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    elif n_chns == 8:
        fig = plt.figure(figsize=(7, 5))
        axes = fig.subplots(nrows=4, ncols=2, sharex=True, sharey=True)

    for i in range(n_chns):
        chn = stn.get_channel(i)
        if i == 0:
            ax = axes[0, 0]
            color = 'red'
            #antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude')
        elif i == 1:
            ax = axes[0, 1]
            color = 'green'
            #antenna_type = 'downward LPDA'
        elif i == 2:
            ax = axes[1, 0]
            color = 'blue'
            #antenna_type = 'downward LPDA'
            ax.set_ylabel('Amplitude')
        elif i == 3:
            ax = axes[1, 1]
            color = 'C1'
            #antenna_type = 'downward LPDA'
        elif i == 4:
            ax = axes[2, 0]
            color = '0.5'
            #antenna_type = 'dipole'
            ax.set_ylabel('Amplitude')
        elif i == 5:
            ax = axes[2, 1]
            color = '0.5'
            #antenna_type = 'upward LPDA'
        elif i == 6:
            ax = axes[3, 0]
            color = '0.5'
            #antenna_type = 'dipole'
            ax.set_ylabel('Amplitude')
            ax.set_xlabel('Frequencies [MHz]')
        elif i == 7:
            ax = axes[3, 1]
            color = '0.5'
            #antenna_type = 'upward LPDA'
            ax.set_xlabel('Frequencies [MHz]')
        ax.set_title(r'channel{}'.format(
            i))
        frequencies = chn.get_frequencies()/units.MHz
        amplitude = np.abs(chn.get_frequency_spectrum())
        max_index = np.argmax(amplitude)
        ax.plot(frequencies, amplitude, color=color, lw=1)

    return fig, axes
