import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.eventWriter
import datetime
import NuRadioReco.detector
import NuRadioReco.utilities
import NuRadioReco.framework.parameters
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import NuRadioReco.modules.channelStopFilter
import glob
import sys
import os

'''
usage: python3 single_template_correlation.py station_id

Reads in .nur format data files and correlate with 1 cosmic-ray template. Save the events to a single .nur file

'''

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('plots/'):
    os.mkdir('plots/') 


station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    files=glob.glob('data/station_32/*')
    template_directory='templates/station_32/'
    output_name='results/single_tplt_station_32.nur'
elif station_id==52:
    used_channels=[4,5,6,7]
    files=glob.glob('data/station_32/*')
    template_directory='templates/station_52/'
    output_name='results/single_tplt_station_52.nur'
else:
    print('usage: python3 single_template_correlation.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0


#load all modules
channelResampler=NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin(debug=False)
channelTemplateCorrelation=NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(template_directory=template_directory)
channelSignalReconstructor=NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
det=NuRadioReco.detector.detector.Detector()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
eventWriter=NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_name,max_file_size=102400)  


data=NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(files)
for event in data.get_events():
    for station in event.get_stations():
        station_time=station.get_station_time()
        det.update(station_time)
        if not(station.get_station_time() > datetime.datetime(2018, 12, 1)) :
            continue
        channelStopFilter.run(event,station,det)
        channelBandPassFilter.run(event, station, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order = 10)
        channelTemplateCorrelation.run(event,station,det,cosmic_ray=True, channels_to_use=used_channels)
        xcorr=station[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations]['cr_avg_xcorr_parallel_crchannels']
        print(f'cr_avg_xcorr_crchannels={xcorr:.2f}')
        channelSignalReconstructor.run(event,station,det)
        eventWriter.run(event)

eventWriter.end()