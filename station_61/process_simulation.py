from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import particleParameters as pp
from NuRadioReco.framework.parameters import generatorAttributes as ga
from NuRadioReco.framework.parameters import showerParameters as swp
import matplotlib.pyplot as plt
import numpy as np
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
import datetime
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.triggerTimeAdjuster

det = detector.Detector()
det.update(datetime.datetime(2018, 12, 15))
nurFile = ['']
template = NuRadioRecoio.NuRadioRecoio(nurFile)
template_directory = ''
used_channels_fit = [0, 1, 2, 3]
output_name = ''

channelTemplateCorrelation = NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(
    template_directory=template_directory)
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_name)
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin()

for evt in template.get_events():
    stn = evt.get_station(61)
    sim_stn = stn.get_sim_station()
    shower = evt.get_first_sim_shower()

    hardwareResponseIncorporator.run(evt, stn, det, sim_to_data=False)
    channelGenericNoiseAdder.run(evt, stn, det, type="rayleigh", amplitude=12.9 *
                                 units.microvolt, min_freq=None, max_freq=500 * units.MHz)
    hardwareResponseIncorporator.run(evt, stn, det, sim_to_data=True)
    channelStopFilter.run(evt, stn, det, prepend=0, append=0)
    channelAddCableDelay.run(evt, stn, det, mode='subtract')
    channelResampler.run(evt, stn, det, sampling_rate=10 *
                         NuRadioReco.utilities.units.GHz)
    channelBandPassFilter.run(evt, stn, det, passband=[
                              80 * units.MHz, 500 * units.MHz], filter_type='butter', order=10)
    channelTemplateCorrelation.run(
        evt, stn, det, cosmic_ray=False, n_templates=1, channels_to_use=used_channels_fit)
    channelSignalReconstructor.run(evt, stn, det)
    channelResampler.run(evt, stn, det, sampling_rate=1 *
                         NuRadioReco.utilities.units.GHz)
    eventWriter.run(evt)
eventWriter.end()
