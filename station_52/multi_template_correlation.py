import os
import sys
import NuRadioReco.detector
import NuRadioReco.utilities
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.framework.parameters
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelStopFilter


'''
usage: python3 multi_template_correlation.py station_id

Reads in output from single_template_correlation.py and correlate events with 1000 cosmic-ray templates. Save output into a single .nur file

'''


station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    files=['results/single_tplt_station_32.nur']
    template_directory='templates/station_32'
    output_name='results/multi_tplt_station_32.nur'

elif station_id==52:
    used_channels=[4,5,6,7]
    files=['results/single_tplt_station_52.nur']
    template_directory='templates/station_52'
    output_name='results/multi_tplt_station_52.nur'

else:
    print('usage: python3 multi_template_correlation.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0

#number of template to use for channelTemplateCorrelation
n_templates=1000


channelResampler=NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin(debug=False)
channelTemplateCorrelation=NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(template_directory=template_directory)
eventWriter=NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_name,max_file_size=102400)
det=NuRadioReco.detector.detector.Detector()


data=NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(files)
for event in data.get_events():
    for station in event.get_stations():
        xcorr=station[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations]['cr_avg_xcorr_parallel_crchannels']

        # do not correlate to 1000 tempaltes if correlation < 0.4 in single template correlation to save runtime
        if xcorr<0.4:
            eventWriter.run(event)
            continue
        print('best correlation with single template was {}'.format(xcorr))
        station_time=station.get_station_time()
        det.update(station_time)
        channelTemplateCorrelation.run(event,station,det,cosmic_ray=True,channels_to_use=used_channels,n_templates=n_templates)
        print('average correlation is %lf'%station[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations]['cr_avg_xcorr_parallel_crchannels'])
        eventWriter.run(event)

nevents=eventWriter.end()
