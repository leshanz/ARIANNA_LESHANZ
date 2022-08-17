from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.utilities import units,  fft
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.framework.parameters
import numpy as np
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import sys


'''
usage: python3 thermal_noise.py station_id

average the spectrum of forced trigger events to get a noise spectrum and save it to a .npy file

The output is used in FullReconstruction.py to add noise to simulated events

'''



station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    nurfiles=['data/station_32/station_32_run_00281.root.nur']
    rms_noise=20 * units.mV
    chn_number=4

elif station_id==52:
    used_channels=[4,5,6,7]
    nurfiles=['data/station_52/station_52_run_00303.root.nur']
    rms_noise=10 * units.mV
    chn_number=8

else:
    print('usage: python3 thermal_noise.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0


det = detector.Detector()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()




power_dict={}
for i in range(chn_number):
    power_dict[i]=np.array([np.complex128(0.0)] * 1281)


data=NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(nurfiles)
counter=0
for event in data.get_events():
    event_id=event.get_id()
    run_number=event.get_run_number()
    station=event.get_station(station_id)
    if not station.has_triggered():
        det.update(station.get_station_time())
        station_time=station.get_station_time()
        channelResampler.run(event, station, det, sampling_rate=10 * units.GHz)
        hardwareResponseIncorporator.run(event,station,det)

        for channel in station.iter_channels():
            sampling_rate=channel.get_sampling_rate()
            trace=channel.get_trace()
            trace[:5]=0
            trace[-5:]=0
            channel.set_trace(trace,sampling_rate)
            channel_id=channel.get_id()
            trace_fft = abs(fft.time2freq(trace, sampling_rate))
            power_dict[channel_id]+=trace_fft
        counter+=1


amp_dict={}
for i in range(chn_number):
    amp_dict[i]=power_dict[i]/len(counter)


if station_id==32:
    np.save('results/station_32_thermal_nosie',[amp_dict[0],amp_dict[1],amp_dict[2],amp_dict[3]])
elif station_id==52:
    np.save('results/station_52_thermal_nosie',[amp_dict[0],amp_dict[1],amp_dict[2],amp_dict[3],amp_dict[4],amp_dict[5],amp_dict[6],amp_dict[7]])


            