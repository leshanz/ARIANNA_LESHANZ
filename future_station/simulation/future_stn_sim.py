import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import os
import sys
import time
import secrets
import argparse
from NuRadioMC.utilities import runner
import argparse
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
import logging
import copy
import yaml

root_seed = secrets.randbits(128)

# initialize detector sim modules
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=200*units.ns)


def task(q, iSim, nu_energy, nu_energy_max, detectordescription, config, output_filename,
         **kwargs):

    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):

            channelBandPassFilter.run(
                evt, station, det, passband=[80*units.MHz, 500*units.MHz], filter_type='butter', order=10)

        def _detector_simulation_trigger(self, evt, station, det):

            trace_copy = {}
            for i in range(8):
                chn = station.get_channel(i)
                trace = chn.get_trace()
                trace_copy[i] = trace
            sampling_rate = chn.get_sampling_rate()

            channelBandPassFilter.run(
                evt, station, det, passband=[80*units.MHz, 180*units.MHz], filter_type='butter', order=10)

            highLowThreshold.run(evt, station, det,
                                 threshold_high=3.94 * 4.9*units.microvolt,
                                 threshold_low=-3.94 * 4.9*units.microvolt,
                                 coinc_window=30*units.ns,
                                 # select the LPDA channels
                                 triggered_channels=[0, 1, 2, 3],
                                 number_concidences=2,  # 2/4 majority logic
                                 trigger_name='2of4_3.94sigma')

            for i in range(8):
                chn = station.get_channel(i)
                chn.set_trace(trace_copy[i], sampling_rate)

    r_max = 2 * units.km
    #r_max = get_max_radius(nu_energy)
    volume = {'fiducial_rmax': r_max,
              'fiducial_rmin': 0 * units.km,
              'fiducial_zmin': -2.7 * units.km,
              'fiducial_zmax': 0
              }

    n_events = 1e4

    input_data = generator.generate_eventlist_cylinder("on-the-fly", n_events, nu_energy, nu_energy_max,
                                                       volume,
                                                       thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                                       phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                                       start_event_id=0,
                                                       flavor=[
                                                           12, -12, 14, -14, 16, -16],
                                                       n_events_per_file=None,
                                                       spectrum='GZK-1',
                                                       deposited=False,
                                                       proposal=False,
                                                       proposal_config='SouthPole',
                                                       start_file_id=0,
                                                       log_level=None,
                                                       proposal_kwargs={},
                                                       max_n_events_batch=n_events,
                                                       write_events=False,
                                                       seed=root_seed + iSim,
                                                       interaction_type="ccnc")

    sim = mySimulation(inputfilename=input_data,
                       outputfilename=output_filename,
                       detectorfile=detectordescription,
                       outputfilenameNuRadioReco=output_filename[:-5]+".nur",
                       config_file=config,
                       default_detector_station=61,
                       file_overwrite=True)
    n_trig = sim.run()

    print(f"simulation pass {iSim} with {n_trig} events", flush=True)
    q.put(n_trig)


if __name__ == "__main__":

    kwargs = {}
    kwargs['detectordescription'] = 'simulation/arianna_detector_db.json'
    kwargs['config'] = 'simulation/config.yaml'
    kwargs['nu_energy'] = 10 ** 17
    kwargs['nu_energy_max'] = 10 ** 19.5
    kwargs['index'] = int(sys.argv[1])

    output_path = ""

    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    class myrunner(runner.NuRadioMCRunner):

        # if required override the get_outputfilename function for a custom output file
        def get_outputfilename(self):
            return os.path.join(self.output_path, f"{self.kwargs['index']:02d}_{self.i_task:06d}.hdf5")

    r = myrunner(20, task, output_path, max_runtime=10*3600, kwargs=kwargs)
    r.run()
