import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import os
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
triggerTimeAdjuster.begin()


def task(q, iSim, nu_energy, nu_energy_max, detectordescription, config, output_filename,
         **kwargs):

    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):

            hardwareResponseIncorporator.run(
                evt, station, det, sim_to_data=True)

        def _detector_simulation_trigger(self, evt, station, det):
            highLowThreshold.run(evt, station, det,
                                 threshold_high=4.3 * self._Vrms,
                                 threshold_low=-4.3 * self._Vrms,
                                 coinc_window=30*units.ns,
                                 # select the LPDA channels
                                 triggered_channels=[0, 1, 2, 3],
                                 number_concidences=2,  # 2/4 majority logic
                                 trigger_name='LPDA_2of4_4.3sigma')

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
    kwargs['detectordescription'] = 'arianna_detector_db.json'
    kwargs['config'] = 'config.yaml'
    kwargs['nu_energy'] = 10 ** 17
    kwargs['nu_energy_max'] = 10 ** 19.5

    output_path = ""

    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    class myrunner(runner.NuRadioMCRunner):

        # if required override the get_outputfilename function for a custom output file
        def get_outputfilename(self):
            # return os.path.join(self.output_path, f"{self.kwargs['index']:06d}_{self.i_task:06d}.hdf5")
            return os.path.join(self.output_path, f"stn61_{self.i_task:06d}.hdf5")

    # start a simulation on 20 cores with a runtime of 2days
    r = myrunner(20, task, output_path, max_runtime=10*3600, kwargs=kwargs)
    r.run()
