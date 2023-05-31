import numpy as np
import operator
import logging
import time

from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma

import optuna

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('GWtunaLivingston.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

class NeedsInvestigatingCallback(object):
    """A callback for Optuna which identifies potential events."""

    def __init__(self, early_stopping_rounds: int, snr_threshold: int, direction: str = "minimize") -> None:
        self.snr_threshold = snr_threshold
        self.early_stopping_rounds = early_stopping_rounds
        
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Goes onto Stocastic Gradient Descent."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._score >= self.snr_threshold:
            if self._iter >= self.early_stopping_rounds:
                study.stop()
                print(study.best_params)

def confirmed_gw_timeseries(event, detector):
    merger = Merger(event)

    # Get the data from the Hanford detector
    strain = merger.strain(detector)

    # Remove the low frequency content and downsample the data to 2048Hz
    strain = resample_to_delta_t(highpass(strain, 15.0), 1.0/2048)

    # Remove 2 seconds of data from both the beginning and end
    conditioned = strain.crop(2, 2)
    
    return conditioned

def confirmed_gw_psd(conditioned):
    psd = conditioned.psd(4)

    psd = interpolate(psd, conditioned.delta_f)

    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
                                      low_frequency_cutoff=15)
    return psd

 
def objective(trial):
    m1 = trial.suggest_float('m1', 1, 100, step=0.000001)
    m2 = trial.suggest_float('m2', 1, 100, step=0.000001)
    #s1 = trial.suggest_float('s1', -0.99, 0.99, step=0.000001)
    #s2 = trial.suggest_float('s2', -0.99, 0.99, step=0.000001)
    #hp, hc = get_td_waveform(approximant="IMRPhenomXAS", mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, delta_t=conditioned.delta_t,f_lower=20)
    conditioned = confirmed_gw_timeseries(event, detector='L1')
    psd = confirmed_gw_psd(conditioned)
    hp, hc = get_td_waveform(approximant="IMRPhenomXAS", mass1=m1, mass2=m2, delta_t=conditioned.delta_t,f_lower=20)
    hp.resize(len(conditioned))
    template = hp.cyclic_time_shift(hp.start_time)
    snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=20)
    snr = snr.crop(4 + 4, 4)
    peak = abs(snr).numpy().argmax()
    snrp = abs(snr[peak])
    return float(snrp)

Events = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729", "GW170809", "GW170814", "GW170817", "GW170818", "GW170823"]


logger.info(f'GWtuna is starting for Livingston')

for event in Events: 
    start_time = time.time()
    optuna.logging.disable_default_handler()
    direction="maximize"
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction=direction)
    needs_to_be_investigated = NeedsInvestigatingCallback(300, snr_threshold=9, direction=direction)
    study.optimize(objective, callbacks=[needs_to_be_investigated], n_trials=1000)
    logger.info(("Time taken", time.time() - start_time))
    logger.info(f'The event is {event} and has the best {study.best_params} with a snr {study.best_value}')