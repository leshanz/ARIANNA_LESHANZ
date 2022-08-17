import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.utilities.units as units
from NuRadioReco.framework.parameters import particleParameters as pp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import ariannaHelper


files = []

events = NuRadioRecoio.NuRadioRecoio(files).get_events()

delta_T_list = []
zenith_list = []
direct = []
refracted = []
reflected = []
weight_list = []
for evt in events:
    stn = evt.get_station(61)
    sim_stn = stn.get_sim_station()
    temp_list = []
    weight = ariannaHelper.get_weight_simulation(evt)
    weight_list.append(weight)
    zenith, azimuth = ariannaHelper.get_arrival_direction_simulation(stn)
    zenith_list.append(zenith/units.deg)
    ray_type_list = []
    for e_field in sim_stn.get_electric_fields_for_channels(channel_ids=[4]):
        temp_list.append(e_field.get_times()[0]/units.ns)
        ray_type_list.append(e_field[efp.ray_path_type])
    delta_T_list.append(np.max(temp_list) - np.min(temp_list))

zenith_list = np.array(zenith_list)
delta_T_list = np.array(delta_T_list)
weight_list = np.array(weight_list)

fig, ax = plt.subplots()
ax.scatter(zenith_list, delta_T_list,
           s=3)
ax.legend()
ax.set_xlabel('zenith [deg]')
ax.set_ylabel('delta T [ns]')
fig.suptitle('stn61 sim 10m dipole delta T from ray tracing')
# fig.savefig('plots/stn61_sim_zenith_deltaT_ray_tracing.png')
plt.show()
