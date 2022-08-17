import numpy as np
import glob
import sys
import astropy
from radiotools import helper as hp
import radiotools.coordinatesystems as coordinatesystems
from NuRadioReco.utilities import units, fft
from NuRadioReco.detector import detector
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.channelLengthAdjuster


'''
usage: python3 FullReconstruction.py station_id

perform a full reconstruction on CoREAS simulation.

Generates simulated events by doing noise adding, hardware response convolving, trigger simulating.

Then perform the same reconstruction percedure as done to data: bandpass filter, arrival direction and polarization reconstruction


'''


def GetSNRs(station):
    '''
    Get a list of SNRs of an event
    '''
    SNRs=[]
    for channel in station.iter_channels():
        if channel.get_id() in used_channels:
            SNRs.append(channel[NuRadioReco.framework.parameters.channelParameters.maximum_amplitude]/rms_noise)
    return SNRs

random_generator = np.random.RandomState()

def add_noise_data(event, station, detector):
    '''
    add measured noise to simulation
    '''
    channels = station.iter_channels()
    for channel in channels:
        channel_id=channel.get_id()
        trace = channel.get_trace()
        sampling_rate = channel.get_sampling_rate()
        n_samples=trace.shape[0]
        frequencies = channel.get_frequencies()
        n_samples_freq = len(frequencies)
        ampl = np.zeros(n_samples_freq)
        ampl= amp_dict[channel_id]
        noise = channelGenericNoiseAdder.add_random_phases(ampl, n_samples) / sampling_rate
        noise=fft.freq2time(noise, sampling_rate, n=n_samples)
        new_trace = trace + noise
        channel.set_trace(new_trace, sampling_rate)

station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    channel_pairs=((0,2),(1,3))
    noise_file='results/station_32_thermal_nosie.npy'
    rms_noise=20 * units.mV
    outputname='results/station_32_simulation.npy'

elif station_id==52:
    used_channels=[4,5,6,7]
    channel_pairs=((4,6),(5,7))
    noise_file='results/station_52_thermal_nosie.npy'
    rms_noise=10 * units.mV
    outputname='results/station_52_simulation.npy'

else:
    print('usage: python3 FullReconstruction.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0

input_file=glob.glob('coreas/')
noise_data = np.load(noise_file,allow_pickle=True,encoding='bytes')

amp_dict={}
for i in range(4):
    amp_dict[i]=noise_data[i] * 10


det = detector.Detector() # detector file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin(input_file, station_id, n_cores=10, max_distance=None)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter =  NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
electricFieldBandPassFilter=NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
channelTemplateCorrelation=NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(template_directory='templates/',)
channelLengthAdjuster=NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin(number_of_samples=2560)




polarization_list=[]
expectation_list=[]
direction_list=[]
SNRs=[]
exp_direction_list=[]
energy_triggered=[]
energy_all=[]
direction_all=[]
signal_ephi=[]
signal_etheta=[]
noise_ephi=[]
noise_etheta=[]
true_polarization_list=[]
amplitude_list=[]
correlation_list=[]



# Loop over all events in file as initialized in readCoRREAS and perform analysis
for evt in readCoREAS.run(detector=det):
    station = evt.get_station(station_id)
    station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
    det.update(station.get_station_time())
    if simulationSelector.run(evt, station.get_sim_station(), det):


        efieldToVoltageConverter.run(evt, station, det)

        channelLengthAdjuster.run(evt,station,det=det)

        add_noise_data(evt,station,det)

        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        triggerSimulator.run(evt, station,det, number_concidences = 2, threshold = 100 *units.mV,triggered_channels=[0,1,2,3])

        zenith_exp=station.get_sim_station()[NuRadioReco.framework.parameters.stationParameters.zenith]
        azimuth_exp=station.get_sim_station()[NuRadioReco.framework.parameters.stationParameters.azimuth]
        direction_all.append((zenith_exp,azimuth_exp))
        energy_all.append(station.get_sim_station()[NuRadioReco.framework.parameters.stationParameters.cr_energy])

        if station.get_trigger('default_simple_threshold').has_triggered():

            ss = station.get_sim_station()
            channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)

            channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order = 10)

            eventTypeIdentifier.run(evt, station, "forced", 'cosmic_ray')

            channelSignalReconstructor.run(evt, station, det)
            SNRs.append(GetSNRs(station))        

            channelTemplateCorrelation.run(evt,station,det,cosmic_ray=True, channels_to_use=used_channels,n_templates=1000)

            hardwareResponseIncorporator.run(evt, station, det)

            correlationDirectionFitter.run(evt, station, det, n_index=1., channel_pairs=channel_pairs)

            voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=used_channels, bandpass=[80*units.MHz, 500*units.MHz], use_MC_direction=False)
            
            electricFieldBandPassFilter.run(evt, ss, det, [80 * units.MHz, 300 * units.MHz])

            electricFieldResampler.run(evt, ss, det, sampling_rate=1 * units.GHz)

            ef = station.get_electric_fields()[0]
            f_theta = np.sum(ef.get_trace()[1] ** 2)
            f_phi = np.sum(ef.get_trace()[2] ** 2)
            polarization = np.arctan2(f_phi ** 0.5, f_theta ** 0.5)
            zenith=station[NuRadioReco.framework.parameters.stationParameters.zenith]
            azimuth=station[NuRadioReco.framework.parameters.stationParameters.azimuth]
            energy=ss[NuRadioReco.framework.parameters.stationParameters.cr_energy]
            amplitude=station[NuRadioReco.framework.parameters.stationParameters.channels_max_amplitude]
            amplitude_list.append(amplitude)
            correlation=station[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations]['cr_avg_xcorr_parallel_crchannels']
            correlation_list.append(correlation)

            bfield_coreas = ss.get_magnetic_field_vector()
            cs = coordinatesystems.cstrafo(zenith, azimuth,magnetic_field_vector=bfield_coreas)
            exp_efield_coreas = hp.get_lorentzforce_vector(zenith, azimuth, bfield_coreas)
            exp_efield_onsky_coreas = cs.transform_from_ground_to_onsky(exp_efield_coreas)
            exp_pol_angle_coreas = abs(np.arctan2(abs(exp_efield_onsky_coreas[2]), abs(exp_efield_onsky_coreas[1])))
            expectation=abs(exp_pol_angle_coreas)
            
            energy_triggered.append(energy)
            polarization_list.append(polarization)
            expectation_list.append(expectation)
            direction_list.append((zenith,azimuth))
            exp_direction=(ss[NuRadioReco.framework.parameters.stationParameters.zenith],ss[NuRadioReco.framework.parameters.stationParameters.azimuth])
            exp_direction_list.append(exp_direction)
            energy_fluence_eTheta = np.sum(ss.get_electric_fields()[0].get_trace()[1]**2)
            energy_fluence_ePhi = np.sum(ss.get_electric_fields()[0].get_trace()[2]**2)
            true_polarization_energy = np.arctan2(energy_fluence_ePhi**0.5,energy_fluence_eTheta**0.5)
            true_polarization_list.append(true_polarization_energy)



np.save(outputname,[polarization_list,expectation_list,direction_list,SNRs,exp_direction_list,energy_all,energy_triggered,direction_all,true_polarization_list,amplitude_list,correlation_list])
print("Finished processing, {} events".format(len(polarization_list)))
