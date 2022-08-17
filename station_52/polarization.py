
import numpy as np
import sys
from NuRadioReco.utilities import units
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.electricFieldBandPassFilter
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.detector import detector
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.electricFieldBandPassFilter

'''
usage: python3 polarization.py station_id

use output from CR_selection and multi_template_correlation.

reconstructs polarization and signal arrival direction for selected events

'''


channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin(debug=False)
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)
voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

station_id=sys.argv[1]


if station_id==32:
    channel_pairs=((0,2),(1,3))
    use_channels=[0,1,2,3]
    files=['results/multi_tplt_station_32.nur']
    cr_evts_file='results/station_32_evts_passing_corr_cut.npy'
    outputname='results/station_32_polarization.npy'

elif station_id==52:
    channel_pairs=((4,6),(5,7))
    use_channels=[4,5,6,7]
    files=['results/multi_tplt_station_52.nur']
    cr_evts_file='results/station_52_evts_passing_corr_cut.npy'
    outputname='results/station_52_polarization.npy'

else:
    print('usage: python3 polarization.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0


det = detector.Detector(source='json',assume_inf=False)
id_runnum_pairs=np.load(cr_evts_file,allow_pickle=True,encoding='bytes')[0]
data=NuRadioRecoio.NuRadioRecoio(files)

direction_list=[]
id_runnum_results=[]
pol_fluence_list=[]
expectation_list=[]


for evt in data.get_events():
    station=evt.get_station(station_id)
    
    if not (evt.get_id(),evt.get_run_number()) in id_runnum_pairs:
        continue
    
    det.update(station.get_station_time())
    eventTypeIdentifier.run(evt, station, "forced", 'cosmic_ray')
    channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 300 * units.MHz], filter_type='rectangular')
    hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
    correlationDirectionFitter.run(evt,station,det,n_index=1.0,channel_pairs=channel_pairs)
    id_runnum_results.append((evt.get_id(),evt.get_run_number()))
    direction_list.append([station.get_parameter(stnp.zenith),station.get_parameter(stnp.azimuth)])
    voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=use_channels, bandpass=[80*units.MHz, 300*units.MHz], use_MC_direction=False)
    electricFieldResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
    electricFieldBandPassFilter.run(evt, station, det, [80 * units.MHz, 300 * units.MHz])
    electricFieldSignalReconstructor.run(evt, station, det)

    ef = station.get_electric_fields()[0]
    f_theta = np.sum(ef.get_trace()[1] ** 2)
    f_phi = np.sum(ef.get_trace()[2] ** 2)
    polarization = np.arctan2(f_phi ** 0.5, f_theta ** 0.5)
    expectation=station.get_electric_fields()[0][NuRadioReco.framework.parameters.electricFieldParameters.polarization_angle_expectation]

    pol_fluence_list.append(polarization)
    expectation_list.append(expectation)

np.save(outputname,[id_runnum_results,direction_list,pol_fluence_list,expectation_list])






