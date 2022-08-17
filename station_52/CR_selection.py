import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp


'''
usage: python3 CR_selection.py station_id

Reads in output from multi_template_correlation.py and apply rate cut and correlation cut
Makes event time VS correlation plot
Outputs a list of event id of events passing the two cut for polarization reconstruction

'''




def ExclusionRegion(x):

    '''
    defines the correlation cut cutline
    '''

    x = np.array([x])
    y = np.zeros_like(x)
    y[np.where(x < 10**2.05)] = 0.4
    y[np.where((x < 10**2.3) & (x > 10**2.05))] = 0.4 + ((0.7 - 0.4) / (2.3 - 2.05)) * (np.log10(x[np.where((x <10** 2.3) & (x > 10**2.05))]) - 2.05)
    y[np.where((x < 10**3.0) & (x > 10**2.3))] = 0.7 + ((0.8 - 0.7) / (3.0 - 2.3)) * (np.log10(x[np.where((x < 10**3.0) & (x >10** 2.3))]) - 2.3)
    y[np.where(x > 10**3.0)] = 0.8

    return y

def CutCriteria(corr, amplitude):

    '''
    applies the correlation cut
    '''

    cut = False
    amplitude =np.array(amplitude)
    if corr < 0.4:
        cut = True
    else:
        if corr < ExclusionRegion(amplitude):
            cut = True
    return cut


def EvtRateDict(events):
    '''
    generate a dictionary of number of events for each run
    '''
    dict={}
    for event in events:
        for station in event.get_stations():

            if not( station.get_station_time() > datetime.datetime(2018, 12, 1) and station.has_triggered()) :
                continue
            seq_num=station.get_ARIANNA_parameter(NuRadioReco.framework.parameters.ARIANNAParameters.seq_num)
            run_number=event.get_run_number()
            xcorr=station.get_parameter(NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations)['cr_avg_xcorr_parallel_crchannels']
            if xcorr>0.4:
                if not (seq_num,run_number) in dict.keys():
                    dict[(seq_num,run_number)]=1
                else:
                    dict[(seq_num,run_number)]+=1
            else:
                if not (seq_num,run_number) in dict.keys():
                    dict[(seq_num,run_number)]=0
    return dict

def EventRate(events,dict):
    '''
    calculate event rate based on the dictionary
    '''
    
    rate=[]
    keys=list(dict.keys())
    n_events=0
    for event in events:
        run_number=event.get_run_number()
        for station in event.get_stations():
            if not( station.get_station_time() > datetime.datetime(2018, 12, 1) and station.has_triggered()) :
                continue
            seq=station.get_ARIANNA_parameter(NuRadioReco.framework.parameters.ARIANNAParameters.seq_num)
            index=keys.index((seq,run_number))
            if index <=12:
                for i in range(24):
                    n_events += dict[keys[i]]
            if index >= len(keys)-13:
                for i in range(24):
                    n_events += dict[keys[-1-i]]
            else:
                for i in range(24):
                    n_events+= dict[keys[index-12 + i]]
            rate.append(n_events)
            n_events=0
    return rate


station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    files=['results/multi_tplt_station_32.nur']
    rms_noise=20 * units.mV

elif station_id==52:
    used_channels=[4,5,6,7]
    files=['results/multi_tplt_station_52.nur']
    rms_noise=10 * units.mV

else:
    print('usage: python3 CR_selection.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0



det = detector.Detector()
correlation_key = 'cr_avg_xcorr_parallel_crchannels'

amplitude=[]
station_time=[]
id_runnum_pair_list=[]
xcorr=[]

data = NuRadioRecoio.NuRadioRecoio(files)
for evt in data.get_events():

    stn=evt.get_station(station_id)

    if not( stn.get_station_time() > datetime.datetime(2018, 12, 1) and stn.has_triggered()) :
        continue

    station_time.append(stn.get_station_time())
    max_cmp_chns=[]
    channel_amp_dict={}
    amplitude.append(stn[stnp.channels_max_amplitude])
    id_runnum_pair_list.append((evt.get_id(),evt.get_run_number()))
    xcorr.append(stn[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations][correlation_key])
    for i in used_channels:
        channel=stn.get_channel(i)
        channel_amp_dict[i]=np.max(np.abs(channel.get_trace()))


station_time=np.array(station_time)
id_runnum_pair_list=np.array(id_runnum_pair_list)
xcorr=np.array(xcorr)
print("{} events".format(len(xcorr)))


amplitude=np.array(amplitude)
xcorr_cut = xcorr > 0.4


events=data.get_events()
dict=EvtRateDict(events)
events=data.get_events()
event_rate_events = EventRate(events,dict)
event_rate_events=np.array(event_rate_events)

rate_cut_threshold = 4

rate_cut = (event_rate_events < rate_cut_threshold)
rate_cut=np.array(rate_cut)

station_time_plot = np.array([x.datetime for x in station_time])

corr_cut = np.array([CutCriteria(x, y) for (x, y) in zip(xcorr[rate_cut], amplitude[rate_cut]/units.mV)])


print(f'#triggered evts={len(xcorr):.0f}')
print(f'#passing rate cut={len(xcorr[rate_cut]):.0f}')
print(f'# passing corr cut={len(xcorr[rate_cut][~corr_cut]):.0f}')

id_runnum_pair_save=[]
for i in id_runnum_pair_list[rate_cut][~corr_cut]:
    id_runnum_pair_save.append((i[0],i[1]))

np.save(f'results/station_{station_id}_evts_passing_corr_cut',[id_runnum_pair_save])

fig,ax=plt.subplots(figsize=(6,4))
ax.text(datetime.datetime(2019,1,7),0.9,'station 32')
ax.set_xlabel('event time')
ax.set_ylabel(r'$\overline{\chi}$')
ax.set_ylim(0,1)
ax.scatter(station_time_plot,xcorr,label=f'triggered events',s=2,color='0.8')
ax.scatter(station_time_plot[rate_cut],xcorr[rate_cut],label=f'events passing rate cut',s=2,color='0.2',marker="^")
ax.legend(loc=4,markerscale=2)
plt.gcf().autofmt_xdate()
fig.tight_layout()
fig.savefig(f'plots/rate_cut_32.png')

plt.show()
