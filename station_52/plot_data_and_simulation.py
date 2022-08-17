
import matplotlib.pyplot  as plt
import numpy as np
import sys
import datetime
import NuRadioReco.utilities.units
import NuRadioReco.detector.detector
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.framework.parameters
from radiotools import stats
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio


'''
usage: plot_data_and_simulation.py station_id

reads in output from FullReconstruction.py and multi_template_correlation.py

Apply zenith cut and SNR cut to data

reweight simulation according to energy flux, zenith distribution, and azimuth distribution

makes amplitude VS correlation plot
makes correlation histogram

'''


def get_sigma68(data,weight):
    '''
    convenience function to get sigma68
    '''

    sig=stats.quantile_1d(data,weight,0.68)
    return str(round(sig,1))

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


def PlotRegion():
    '''
    plot the correlation cut cutline
    '''
    x = np.arange(10**1.,10** 4.0,10** 0.01)
    y = ExclusionRegion(x)[0]
    plt.plot(x, y, c='k', linewidth=2, linestyle='--')

def get_mean(data,weight):
    '''
    convenience to get mean and stadard deviation
    '''
    mean,variance=stats.mean_and_variance(data,weight)
    return [str(round(mean,2)),str(round(variance**0.5,2))]



station_id=sys.argv[1]

if station_id==32:
    used_channels=[0,1,2,3]
    files=['results/multi_tplt_station_32.nur']
    pol_file='results/station_32_polarization.npy'
    rms_noise=20 * units.mV

elif station_id==52:
    used_channels=[4,5,6,7]
    files=['results/multi_tplt_station_52.nur']
    pol_file='results/station_52_polarization.npy'
    rms_noise=10 * units.mV

else:
    print('usage: python3 plot_data_and_simulation.py station_id \nOnly station 32 & 52 available')
    a= 1./ 0



det = detector.Detector()
data = NuRadioRecoio.NuRadioRecoio(files)
pol_data=np.load(pol_file,allow_pickle=True,encoding='bytes')

SNR_threshold=4.5
energy_threshold=17.5
zen_threshold1= 60 * units.deg
zen_threshold2= 70 * units.deg
zen_threshold= 40 * units.deg
rate_cut_threshold = 4

correlation_list=[]
amplitude_list=[]
station_time_list=[]
id_runnum_pair_list=[]

station_time_list_all=[]

for evt in data.get_events():
    stn=evt.get_station(station_id)
    station_time=stn.get_station_time()
    station_time_list_all.append(station_time)
    if not( stn.get_station_time() > datetime.datetime(2018, 12, 1) and stn.has_triggered()) :
        continue
    correlation_list.append(stn[NuRadioReco.framework.parameters.stationParameters.cr_xcorrelations]['cr_avg_xcorr_parallel_crchannels'])
    amplitude_list.append(stn[NuRadioReco.framework.parameters.stationParameters.channels_max_amplitude])
    station_time_list.append(stn.get_station_time().datetime)
    id_runnum_pair_list.append((evt.get_id(),evt.get_run_number()))



station_time_list=np.array(station_time_list)
id_runnum_pair_list=np.array(id_runnum_pair_list)
correlation_list=np.array(correlation_list)
station_time_list_all=np.array(station_time_list_all)
amplitude_list=np.array(amplitude_list)


events=data.get_events()
dict=EvtRateDict(events)
events=data.get_events()
event_rate_events = EventRate(events,dict)
event_rate_events=np.array(event_rate_events)


rate_cut = event_rate_events < rate_cut_threshold

amp_corr_cut = np.array([CutCriteria(x, y) for (x, y) in zip(correlation_list[rate_cut], amplitude_list[rate_cut]/units.mV)])

difference_list=[]
azimuth_list=[]
zenith_list=[]
direction_list=[]

event_list=[]
for i in id_runnum_pair_list[rate_cut][~amp_corr_cut]:
    event_list.append((i[0],i[1]))

id_runnum_results=[]
for i in pol_data[0]:
    id_runnum_results.append((i[0],i[1]))

SNR_list=[]
for evt in data.get_events():
    evt_id=evt.get_id()
    run_num=evt.get_run_number()
    channel_amp_dict={}
    if (evt_id,run_num) in event_list and (evt_id,run_num) in id_runnum_results:
        stn=evt.get_station(station_id)
        
        if not( stn.get_station_time() > datetime.datetime(2018, 12, 1) and stn.has_triggered()) :
            continue
        for i in used_channels:
            channel=stn.get_channel(i)
            channel_amp_dict[i]=np.max(np.abs(channel.get_trace()))
        if station_id==32:
            min_snr=min([channel_amp_dict[0]/rms_noise,channel_amp_dict[1]/rms_noise,channel_amp_dict[2]/rms_noise,channel_amp_dict[3]/rms_noise])
        elif station_id==52:
            min_snr=min([channel_amp_dict[4]/rms_noise,channel_amp_dict[5]/rms_noise,channel_amp_dict[6]/rms_noise,channel_amp_dict[7]/rms_noise])
        SNR_list.append(min_snr)


direction_list=pol_data[1]
evt_cut=[]
for i in range(len(direction_list)):
    if id_runnum_results[i] in event_list:
        zenith_list.append(direction_list[i][0])
        azimuth_list.append(direction_list[i][1])
        evt_cut.append(True)
    else:
        evt_cut.append(False)


polarization_list=np.rad2deg([float(i) for i in pol_data[2]])
expectation_list=np.rad2deg([float(i) for i in pol_data[3]])
difference_list=abs(polarization_list-expectation_list)

zenith_list=np.array(zenith_list)
azimuth_list=np.array(azimuth_list)
difference_list=np.array(difference_list)

SNR_list=np.array(SNR_list)


evt_mask=[]
for i in range(len(id_runnum_pair_list)):
    if (id_runnum_pair_list[i][0],id_runnum_pair_list[i][1]) in id_runnum_results:
        evt_mask.append(True)
    else:
        evt_mask.append(False)


id_runnum_results=np.array(id_runnum_results)

zen_cut=zenith_list > zen_threshold
SNR_cut= SNR_list[zen_cut] > SNR_threshold


zen_cut1= zenith_list[zen_cut][SNR_cut] > zen_threshold1

zen_cut2=zenith_list[zen_cut][SNR_cut] > zen_threshold2


print(f'#triggered evts={len(correlation_list):.0f}')
print(f'#passing rate cut={len(correlation_list[rate_cut]):.0f}')
print(f'# passing corr cut={len(correlation_list[rate_cut][~amp_corr_cut]):.0f}')
print(f'#zenith>40deg={len(difference_list[evt_cut][zen_cut]):.0f}')
print(f'#passing SNR cut={len(difference_list[evt_cut][zen_cut][SNR_cut]):.0f}')
print(f'#zenith>60deg={len(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut1]):.0f}')
print(f'#zenith>70deg={len(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut2]):.0f}')

bins_list=range(0,90,10)
fig,ax=plt.subplots(figsize=(6,4))
fig.suptitle(r'polarization $\Delta P$')
ax.set_xlabel(r'$\Delta P$ [deg]')
ax.set_ylabel('counts')
ax.hist(difference_list[evt_cut][zen_cut][SNR_cut],linewidth=2,bins=bins_list,ls='--',histtype='step',label=f'zenith>40deg #evts={len(difference_list[evt_cut][zen_cut][SNR_cut]):.0f} sigma68=' + get_sigma68(difference_list[evt_cut][zen_cut][SNR_cut]))
ax.hist(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut1],linewidth=2,bins=bins_list,ls='--',histtype='step',label=f'zenith>60deg #evts={len(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut1]):.0f} sigma68=' + get_sigma68(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut1]))
ax.hist(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut2],linewidth=4,bins=bins_list,histtype='step',color='green',label=f'zenith>70deg #evts={len(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut2]):.0f} sigma68=' + get_sigma68(difference_list[evt_cut][zen_cut][SNR_cut][zen_cut2]))
ax.legend()
ax.set_xticks(bins_list)
fig.tight_layout()
fig.savefig(f'plots/station_{station_id}_polarization.png')


def process_simu(simu_file):

    simu_data = np.load(simu_file,allow_pickle=True,encoding='bytes')
    pol=[float(ele) for ele in simu_data[0]]
    simu_polData=np.rad2deg(np.array(pol))

    simu_energy_all=simu_data[5]
    simu_energy_triggered=simu_data[6]
    simu_SNRs=simu_data[3]

    expected_simu=np.rad2deg(np.array([float(ele) for ele in simu_data[1]]))

    simu_difference=np.abs(np.array(expected_simu-simu_polData))

    true_polarization_energy=np.rad2deg(np.array(simu_data[8]))
    diff_reco_energy=np.abs(np.array(true_polarization_energy-simu_polData))
    diff_geo_energy=np.abs(np.array(true_polarization_energy-expected_simu))


    amp=simu_data[9]
    corr=simu_data[10]
    simu_direction=simu_data[7]
    triggered_simu_direction=simu_data[2]

    simu_zen=np.rad2deg(np.array([ele[0] for ele in simu_direction]))
    simu_azi=np.rad2deg(np.array([ele[1] for ele in simu_direction]))

    triggered_simu_zen=np.rad2deg(np.array([ele[0] for ele in triggered_simu_direction]))
    triggered_simu_azi=np.rad2deg(np.array([ele[1] for ele in triggered_simu_direction]))

    true_direction_list=simu_data[4]
    true_triggered_simu_zen=np.rad2deg(np.array([ele[0] for ele in true_direction_list]))
    true_triggered_simu_azi=np.rad2deg(np.array([ele[1] for ele in true_direction_list]))
    zen_diff=abs(true_triggered_simu_zen-triggered_simu_zen)
    azi_diff=abs(triggered_simu_azi-true_triggered_simu_azi)



    #calculate normalization factor for energy

    fit_distribution=[]
    step=0.25
    length=int((20-17)/step)
    E_i=10**17
    E_f=10**(17+step)
    c=20000/(E_i**-2-E_f**-2)
    for i in range(length-1):
        E_i=10**(17+i*step)
        E_f=10**(17+(i+1)*step)
        N=0.5*c*(E_i**-2-E_f**-2)
        fit_distribution.append(N)
    bin_height={}

    bin_boundary=[10**(17+i*step) for i in range(length)]

    for counter in range(len(bin_boundary)-1):
        bin_height[counter]=0
        for j in simu_energy_all:
            if bin_boundary[counter]<=j<bin_boundary[counter+1]:
                bin_height[counter]+=1

    i_list=[10**(17+(i+0.5)*step) for i in range(length-1)]

    multipliers={}
    for i in range(len(i_list)):
        expected=fit_distribution[i]
        real=bin_height[i]
        multiplier=expected/real
        multipliers[17+i*step]=multiplier

    #normalize simulation distribution by energy

    energy_weights=[]
    selected_diff=[]
    selected_energy=[]
    selected_snrs=[]
    selected_triggered_azi=[]
    selected_azi=[]
    selected_simu_polData=[]
    selected_expected_simu=[]
    selected_zen_diff=[]
    selected_azi_diff=[]
    selected_triggered_simu_zen=[]
    selected_diff_reco_energy=[]
    selected_diff_geo_energy=[]
    selected_true=[]
    selected_expected_simu=[]
    selected_amp=[]
    selected_corr=[]

    for i in range(len(simu_difference)):
        energy=simu_energy_triggered[i]
        if not energy_threshold<=np.log10(energy)<(20-step):
            continue
        selected_diff.append(simu_difference[i])
        selected_energy.append(simu_energy_triggered[i])
        selected_snrs.append(simu_SNRs[i])
        selected_triggered_azi.append(triggered_simu_azi[i])
        selected_azi.append(simu_azi[i])
        selected_simu_polData.append(simu_polData[i])
        selected_expected_simu.append(expected_simu[i])
        selected_zen_diff.append(zen_diff[i])
        selected_azi_diff.append(azi_diff[i])
        selected_triggered_simu_zen.append(triggered_simu_zen[i])
        selected_diff_reco_energy.append(diff_reco_energy[i])
        selected_diff_geo_energy.append(diff_geo_energy[i])
        selected_true.append(true_polarization_energy[i])

        selected_amp.append(amp[i])
        selected_corr.append(corr[i])

        for j in [17+i*step for i in range(length-1)]:
            if j<=np.log10(energy)<j+step:
                multi=multipliers[j]
                energy_weights.append(multi)
    energy_weights=np.array(energy_weights)

    selected_simu_polData=np.array(selected_simu_polData)
    selected_expected_simu=np.array(selected_expected_simu)
    selected_zen_diff=np.array(selected_zen_diff)
    selected_azi_diff=np.array(selected_azi_diff)
    selected_triggered_simu_zen=np.array(selected_triggered_simu_zen)
    selected_azi=np.array(selected_azi)
    selected_triggered_azi=np.array(selected_triggered_azi)

    selected_diff_reco_energy=np.array(selected_diff_reco_energy)
    selected_diff_geo_energy=np.array(selected_diff_geo_energy)

    selected_amp=np.array(selected_amp)
    selected_corr=np.array(selected_corr)


    #calculate normalization factor for azimuth and zenith

    C=1000.0
    azi_multipliers={}
    azi_bin_height={}
    azi_step=20

    zen_multipliers={}
    zen_bin_height={}
    zen_step=5

    selected_energy_all=[]
    energy_weights_all=[]
    selected_azi_all=[]
    selected_zen_all=[]
    for c,i in enumerate(simu_energy_all):
        if not 17.0<=np.log10(i)<(20-step):
            continue
        selected_energy_all.append(i)
        selected_azi_all.append(simu_azi[c])
        selected_zen_all.append(simu_zen[c])
        for j in [17+i*step for i in range(length-1)]:
            if j<=np.log10(i)<j+step:
                multi=multipliers[j]
                energy_weights_all.append(multi)    
            

    counts,bins=np.histogram(selected_azi_all,weights=energy_weights_all,bins=[i for i in range(0,360+azi_step,azi_step)])

    zen_counts,zen_bins=np.histogram(selected_zen_all,weights=energy_weights_all,bins=[i for i in range(0,90+zen_step,zen_step)])


    zen_fit_distribution=[]
    length=int(90/zen_step)
    c=20000
    for i in range(length):
        zen_i=i*zen_step
        zen_f=(i+1)*zen_step
        N=c*np.sin(np.deg2rad(zen_i))* np.deg2rad(zen_f-zen_i)
        zen_fit_distribution.append(N)
    zen_bin_height={}


    for i in range(len(counts)):
        azi_bin_height[azi_step*i]=counts[i]

    for i in range(len(zen_counts)):
        zen_bin_height[zen_step*i]=zen_counts[i]

    for c,i in enumerate(azi_bin_height):
        azi_multipliers[i]=(1/azi_bin_height[i])

    zen_multipliers={}
    i_list=[i * zen_step for i in range(length)]

    for i in range(len(i_list)):
        expected=zen_fit_distribution[i]
        real=zen_bin_height[i*zen_step]
        multiplier=expected/real
        zen_multipliers[i*zen_step]=multiplier


    #normalize by azimuthal and zenith distribution

    azi_weights=[]
    for i in range(len(selected_diff)):
        azimuth=selected_azi_all[i]
        if azimuth>=360.0:
            azi_weights.append(azi_multipliers[360-azi_step])
        for j in range(0,360,azi_step):
            if j<=azimuth<j+azi_step:
                azi_weights.append(azi_multipliers[j])

    zen_weights=[]
    for i in range(len(selected_diff)):
        zenith=selected_zen_all[i]
        if zenith>=90.0:
            zen_weights.append(zen_multipliers[90-zen_step])
        for j in range(0,90,zen_step):
            if j<=zenith<j+zen_step:
                zen_weights.append(zen_multipliers[j])



    combined_weights=[]

    for i in range(len(energy_weights)):
        combined_weights.append(1000*energy_weights[i]*azi_weights[i]*zen_weights[i])


    selected_energy=np.array(selected_energy)
    selected_diff=np.array(selected_diff)
    selected_snrs=np.array(selected_snrs)

    selected_true=np.array(selected_true)

    energy_weights=np.array(energy_weights)
    combined_weights=np.array(combined_weights)

    selected_simu_SNRs=selected_snrs[mask20]

    min_simu_snr=np.array([min(i) for i in selected_simu_SNRs])
    selected_simu_snr_mask=min_simu_snr>=SNR_threshold

    selected_zen_mask=selected_triggered_simu_zen[mask20][selected_simu_snr_mask]>=zen_threshold/units.deg



    return selected_diff, mask20, selected_simu_snr_mask, combined_weights, selected_zen_diff, selected_azi_diff ,energy_weights ,selected_triggered_simu_zen ,selected_zen_mask,selected_diff_reco_energy ,selected_diff_geo_energy ,selected_true , selected_expected_simu,selected_triggered_azi, selected_amp,selected_corr


selected_diff, mask20, selected_simu_snr_mask, combined_weights,zen_diff, azi_diff ,energy_weights ,selected_triggered_simu_zen,selected_zen_mask,selected_diff_reco_energy ,selected_diff_geo_energy,selected_true ,selected_expected_simu,selected_triggered_azi,selected_amp,selected_corr =process_simu(simu_file='results/coreas/stn32_coreas_pol_amp_corr_simu_dire.npy')


bin_list=range(0,20,1)
fig,ax=plt.subplots(figsize=(6,4))
fig.suptitle(f'station_{station_id} simulation')
ax.set_xlabel('polarization error [deg]')
ax.set_ylabel('counts')
ax.hist(selected_diff[selected_zen_mask][selected_simu_snr_mask],bins=bin_list,linestyle='--',linewidth=2,histtype='step',weights=combined_weights[selected_zen_mask][selected_simu_snr_mask],label=r'$zenith>40^{\degree}$ '+r'$\sigma_{68\%}=$'+sig68+r'$^{\degree}$')

zen_threshold = 60 * units.deg
selected_diff, mask20, selected_simu_snr_mask, combined_weights,zen_diff, azi_diff ,energy_weights ,selected_triggered_simu_zen,selected_zen_mask,selected_diff_reco_energy ,selected_diff_geo_energy,selected_true ,selected_a_list,selected_expected_simu,selected_triggered_azi,selected_angle_with_b,sig68,selected_amp,selected_corr =process_simu(simu_file='results/coreas/coreas_pol_amp_corr_data_noise.npy')
ax.hist(selected_diff[selected_zen_mask][selected_simu_snr_mask],bins=bin_list,color='red',linestyle='-.',linewidth=3,histtype='step',weights=combined_weights[selected_zen_mask][selected_simu_snr_mask],label=r'$zenith>60^{\degree}$ '+r'$\sigma_{68\%}=$'+sig68+r'$^{\degree}$')

zen_threshold = 70 * units.deg
selected_diff, mask20, selected_simu_snr_mask, combined_weights,zen_diff, azi_diff ,energy_weights ,selected_triggered_simu_zen,selected_zen_mask,selected_diff_reco_energy ,selected_diff_geo_energy,selected_true ,selected_a_list,selected_expected_simu,selected_triggered_azi,selected_angle_with_b,sig68,selected_amp,selected_corr =process_simu(simu_file='results/coreas/coreas_pol_amp_corr_data_noise.npy')
ax.hist(selected_diff[selected_zen_mask][selected_simu_snr_mask],bins=bin_list,color='green',linewidth=5,histtype='step',weights=combined_weights[selected_zen_mask][selected_simu_snr_mask],label=r'$zenith>70^{\degree}$ '+r'$\sigma_{68\%}=$'+sig68+r'$^{\degree}$')

ax.legend()
ax.set_xticks(bin_list)
fig.tight_layout()
fig.savefig(f'plots/station_{station_id}_simulation_polarization.png')


color_list=[np.amin(combined_weights)+i*(np.log10(np.amax(combined_weights))-0)/len(selected_amp) for i in range(len(selected_amp)) ]
fig,ax=plt.subplots(figsize=(6,6))
fig.suptitle(f'station {station_id}')
ax.set_xlabel('Max amplitude [mV]')
ax.set_ylabel('correlation')
ax.scatter(amplitude_list/ units.mV,correlation_list,label='triggered events',s=2,color='0.8')
plot=ax.scatter(selected_amp/units.mV,selected_corr,c=color_list,cmap=('viridis'),alpha=0.4,s=7,label='simulation')
cbar = fig.colorbar(plot)
ax.set_ylim(0,1)
ax.set_xlim(10**1,10**4)
ax.semilogx(True)
PlotRegion()
cbar.set_label('log10(Weighting multiplier)')
cbar.set_ticks([np.amin(color_list)+i/4*(np.amax(color_list)-np.amin(color_list))for i in range(5)])
cbar.set_ticklabels([round(np.amin(color_list)+i/4*(np.amax(color_list)-np.amin(color_list)),1)for i in range(5)])
ax.scatter(amplitude_list[rate_cut]/ units.mV,correlation_list[rate_cut],label='events passing rate cut',s=2,color='0.2')
ax.scatter(amplitude_list[rate_cut][~amp_corr_cut]/ units.mV,correlation_list[rate_cut][~amp_corr_cut],marker='^',label=f'CR candidates: {len(amplitude_list[rate_cut][~amp_corr_cut]):.0f}',s=12,color='red')
fig.tight_layout()
fig.savefig(f'plots/station_{station_id}_corr_52.png')


bin_list=[0.05*i for i in range(20)]
fig,ax=plt.subplots(figsize=(6,4))
ax.set_xlabel(r'$\overline{\chi}$')
ax.set_ylabel('Normalized counts')
ax.hist(selected_corr,weights=combined_weights,density=True,lw=3,histtype='step',ls='--',color='0.4',bins=bin_list,label=r'simulation: $\mu$='+get_mean(selected_corr,combined_weights)[0])
ax.hist(correlation_list[rate_cut][~amp_corr_cut],density=True,lw=4,histtype='step',bins=bin_list,color='red',label=r'data: $\mu$='+get_mean(correlation_list[rate_cut][~amp_corr_cut],[1] * len(correlation_list[rate_cut][~amp_corr_cut]))[0])
ax.set_xticks([0.1*i for i in range(10)])
ax.legend(loc=2)
ax.text(0.0,2.7,f'station {station_id}')
fig.tight_layout()
fig.savefig(f'plots/station_{station_id}_correlation.png')


plt.show()