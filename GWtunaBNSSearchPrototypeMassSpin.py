#!/usr/bin/env python

# Copyright (C) 2024 Susanna M. Green and Andrew P. Lundgren 

'''GWtuna Search Prototype for Binary Neutron Star Mergers'''

import numpy as np
import operator
import logging
import time

import optuna

import pandas as pd

import jax
import optax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import pycbc.noise
from pycbc.psd.estimate import inverse_spectrum_truncation

from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomD

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('GWtunaMassSpinO4TPESampler1000CmaEsampler900050ipopincpopsize2Callback500Final.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

'''Define the Output file'''
#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/gwtuna/'

OUTPUT_FILE = DATA_DIR+'GWtunaMassSpinO4TPESampler1000CmaEsampler900050ipopincpopsize2Callback500FinalRecoveredSNR.csv'
OUTPUT_FILE2 = DATA_DIR+'GWtunaMassSpinO4TPESampler1000CmaEsampler900050ipopincpopsize2Callback500FinalFailed.csv'

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
        """Stop TPE and start with second trial."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1


        if self._score <= self.snr_threshold:
            if self._iter >= self.early_stopping_rounds:
                study.stop()
                TPE_time = time.time() - TPE_starttime
                logger.info(f'The Stopping algorthim has curtailed the TPE search in {TPE_time} seconds')
                logger.info(f'because SNR has not improved and the SNR thereshold has not been reached')
                logger.info(f'TPE has found the best SNR to be {study.best_value} in {len(study.trials)} trials')
                gwtuna_actual_predicted_timings_failed.append([m1, m2, chi1, chi2, actual_snr_peak, 
                      study.best_params['$M_{1}$'], study.best_params['$M_{2}$'],
                      study.best_params['$\\chi_{1}$'], study.best_params['$\\chi_{2}$'], 
                      float(study.best_value), TPE_time])
    
                
        if self._score >= self.snr_threshold:
            study.stop()
            TPE_time = time.time() - TPE_starttime
            logger.info(f'The Stopping algorthim has curtailed the TPE search')
            logger.info(f"TPESampler found an snr {float(study.best_value)} so snr threshold has reached")
            logger.info(f'CMA-ES will now start.')
            study2 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(restart_strategy='ipop', popsize=50, inc_popsize=2, lr_adapt=False), direction="maximize")
            CMAES_starttime = time.time()
            study2.optimize(optuna_objective, n_trials=9000)
            CMAES_time = time.time() - CMAES_starttime 
            logger.info(f"CmaEsSampler found an snr in {CMAES_time} seconds")
            logger.info(f"CmaEsSampler found an snr {float(study2.best_value)}")
            gwtuna_actual_predicted_timings.append([m1, m2, chi1, chi2, actual_snr_peak, 
                      study.best_params['$M_{1}$'], study.best_params['$M_{2}$'],
                      study.best_params['$\\chi_{1}$'], study.best_params['$\\chi_{2}$'], 
                      float(study.best_value), TPE_time, CMAES_time])
    

# Define sigma squared function 
def sigma_squared_func(delta_freq, invpsd, template):
    weighted_inner = jnp.sum(template*jnp.conj(template)*invpsd)
    h_norm = 4*delta_freq
    sigma_squared = jnp.real(weighted_inner)*h_norm
    return sigma_squared

def injection(constrained_freqs, dynfac, constrained_invpsd, delta_freq, O4_psd, tsamples, delta_t, signal_duration, kmin, kmax, m1, m2, chi1, chi2, tc, des_snr):
    '''Create the Injection Template with a Specified SNR and a chosen timeshift in the Frequency Domain'''
    phic = 1.3
    dist_mpc = 100 # Distance to source in Mpc
    inclination = 1.5 # Inclination Angle
    polarization_angle = 0 # Polarization angle

    f_ref = 20 

    m_chirp, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

    params = jnp.array([m_chirp, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])

    template, _ = IMRPhenomD.gen_IMRPhenomD_hphc(constrained_freqs, params, f_ref)

    desired_snr = des_snr # The SNR we want to compute  = template*dynfac #This converts to sa computer freindly strain

    sigma = sigma_squared_func(delta_freq, constrained_invpsd, template)**(1/2)
    template = (template/sigma)*desired_snr

    '''Jax: Create the Injection Template with a Specified SNR and a chosen timeshift in the Time Domain'''
    blank_space = jnp.zeros(freq_len, dtype=complex) #blank_space = jnp.zeros(int(sampling_rate*signal_duration), dtype=complex) #generated neg and positive so this won't work wih real data
    template_blank_space = blank_space.at[kmin:kmax].set(template)
    time_template = jnp.fft.irfft(template_blank_space) #ifft the template to get in the time domain 
    time_template *= len(time_template) # need this as numpy spits out things that doesn't make physical sense
    time_template *= delta_freq 

    '''Create an Instance in Noise'''
    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, O4_psd, seed=35)

    '''Injection'''
    time_injection = time_template + jnp.array(ts) #Add the noise and the template in the time domain
    freq_injection = jnp.fft.rfft(time_injection)
    freq_injection *= delta_t 

    freq_injection = freq_injection[kmin:kmax]
    return freq_injection

def snrp(constrained_freqs, dynfac, constrained_invpsd, delta_freq, sampling_rate, signal_duration, kmin, kmax, fcore, m1, m2, chi1, chi2):
    '''Template'''
    tc = 0.0 # Time of coalescence in seconds
    phic = 1.3
    dist_mpc = 100 # Distance to source in Mpc
    inclination = 1.5 # Inclination Angle
    polarization_angle = 0 # Polarization angle

    f_ref = 20 

    m_chirp, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

    params = jnp.array([m_chirp, eta, chi1, chi2, dist_mpc, tc, phic, inclination, polarization_angle])

    template, _ = IMRPhenomD.gen_IMRPhenomD_hphc(constrained_freqs, params, f_ref)

    template = template*dynfac #This converts to sa computer freindly strain

    '''Sigma Squared'''
    sigma_squared = sigma_squared_func(delta_freq, constrained_invpsd, template)

    '''Matched_filter'''
    workspace = jnp.zeros(int(sampling_rate*signal_duration), dtype=complex) 
    result_fft = fcore * jnp.conjugate(template) 
    workspace = workspace.at[kmin:kmax].set(result_fft)
    result = jnp.fft.ifft(workspace)
    result *= len(result) 

    '''SNR'''
    norm = 4*delta_freq / jnp.sqrt(sigma_squared)
    snr = result*norm
    snr_min = int((8+150)*sampling_rate) 
    snr_max = len(snr)-int((8)*sampling_rate)
    snr = snr[snr_min:snr_max]
    peak = jnp.argmax(jnp.absolute(snr))
    snrp = jnp.absolute(snr[peak])
    return(snrp)

def optuna_objective(trial):
    m1 = trial.suggest_float('$M_{1}$', MIN_MASS, MAX_MASS)
    m2 = trial.suggest_float('$M_{2}$', MIN_MASS, MAX_MASS)
    chi1 = trial.suggest_float('$\chi_{1}$', MIN_SPIN, MAX_SPIN)
    chi2 =  trial.suggest_float('$\chi_{2}$', MIN_SPIN, MAX_SPIN)
    return my_snrp_jit(fcore, m1, m2, chi1, chi2)

'''Define the Sampling parameters'''
dynfac = 1.0e23
sampling_rate = 2048
nyquist_freq = sampling_rate//2
signal_duration = 512.0 #256
delta_freq = 1/signal_duration
freq_len = int(nyquist_freq / delta_freq)+1 
max_filter_len = 16
delta_t = 1.0 / sampling_rate
tsamples = int(signal_duration / delta_t)

psd_low_freq_cut_off = 18.0 
low_freq_cut_off = 20
high_frq_cut_off = 900
kmin, kmax = int(low_freq_cut_off*signal_duration), int(high_frq_cut_off*signal_duration)

'''Define the Frequency'''
freqs = jnp.linspace(0, int(nyquist_freq), freq_len)
constrained_freqs = freqs[kmin:kmax]

'''Define the Power Spectral Density (PSD)'''
O4_psd = pycbc.psd.read.from_txt("aligo_O4low.txt", freq_len, delta_freq, psd_low_freq_cut_off, is_asd_file=True)*dynfac**2

'''Calculate the Inverse Spectrum Truncation'''
max_filter_samples = max_filter_len*sampling_rate
psd = inverse_spectrum_truncation(O4_psd, max_filter_len=max_filter_samples, low_frequency_cutoff=18) 
constrained_psd = psd[kmin:kmax]

'''Calculate the Inverse PSD'''
invpsd = jnp.array(1/psd)
constrained_invpsd = invpsd[kmin:kmax]

'''Define the parameters for injections'''
NOM_INJECTIONS = 5000
MIN_MASS = 1.0
MAX_MASS = 2.0
MIN_SPIN = -0.05
MAX_SPIN = 0.05
LOW_DESIRED_SNR = 6
HIGH_DESIRED_SNR = 20
LOW_TC = -40.0 
HIGH_TC = -20.0 
injections_mass1 = np.random.uniform(MIN_MASS, MAX_MASS, size=NOM_INJECTIONS)
injections_mass2 = np.random.uniform(MIN_MASS, MAX_MASS, size=NOM_INJECTIONS)
injections_spin1 = np.random.uniform(MIN_SPIN, MAX_SPIN, size=NOM_INJECTIONS)
injections_spin2 = np.random.uniform(MIN_SPIN, MAX_SPIN, size=NOM_INJECTIONS)
injections_tc = np.random.uniform(LOW_TC, HIGH_TC, size=NOM_INJECTIONS)
desired_x = np.random.uniform((LOW_DESIRED_SNR)**(-3), (HIGH_DESIRED_SNR)**(-3), size=NOM_INJECTIONS) # 6**(-3), 20**(-3)
desired_snr = desired_x**(-(1/3))

my_injection = partial(injection, constrained_freqs, dynfac, constrained_invpsd, delta_freq, O4_psd, tsamples, delta_t, signal_duration, kmin, kmax)
my_injection_jit = jax.jit(my_injection)                       
my_snrp = partial(snrp, constrained_freqs, dynfac, constrained_invpsd, delta_freq, sampling_rate, signal_duration, kmin, kmax)
my_snrp_jit = jax.jit(my_snrp)     
    
logger.info(f"Running GWtuna")
start_time = time.time()

gwtuna_actual_predicted_timings = []
gwtuna_actual_predicted_timings_failed = []

for m1, m2, chi1, chi2, tc, des_snr in zip(injections_mass1, injections_mass2, injections_spin1, injections_spin2, injections_tc, desired_snr):
    logger.info(f"Creating injection")
    '''Create Injection'''#Jax.scan... jax.lax.scan(). Write one look of your for loop as a function and then use scan. What you are scanning over  
    freq_injection = my_injection_jit(m1, m2, chi1, chi2, tc, des_snr)

    '''Define fcore'''
    fcore = constrained_invpsd*freq_injection

    '''Calculate actual SNR'''
    actual_snr_peak = my_snrp_jit(fcore, m1, m2, chi1, chi2)
    logger.info(f"Actual snr {float(actual_snr_peak)}")

    optuna.logging.disable_default_handler()
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
    needs_to_be_investigated = NeedsInvestigatingCallback(500, snr_threshold=6, direction="maximize")
    TPE_starttime = time.time()
    study.optimize(optuna_objective, callbacks=[needs_to_be_investigated], n_trials=1000)

logger.info(f"Finished GWtuna")
logger.info("Time taken %s", time.time() - start_time)
logger.info("Time taken for each injection is %s", (time.time() - start_time)/NOM_INJECTIONS)

gwtuna_actual_predicted_timings = np.array(gwtuna_actual_predicted_timings)

GWtunaParamsRecovery =  pd.DataFrame(data=(gwtuna_actual_predicted_timings), columns=['mass1', 'mass2', 'chi1', 'chi2', 'snr', 
                                                                              'predicted_mass1', 'predicted_mass2', 
                                                                              'predicted_spin1', 'predicted_spin2', 'predicted_snr', 'TPE', 'CMAES'])
GWtunaParamsRecovery.to_csv(OUTPUT_FILE, index = False)


gwtuna_actual_predicted_timings_failed = np.array(gwtuna_actual_predicted_timings_failed)

if np.shape(gwtuna_actual_predicted_timings_failed)[0]==0: 
    pass
else: 
    GWtunaParamsRecovery =  pd.DataFrame(data=(gwtuna_actual_predicted_timings_failed), columns=['mass1', 'mass2', 'chi1', 'chi2', 'snr', 
                                                                            'predicted_mass1', 'predicted_mass2', 
                                                                            'predicted_spin1', 'predicted_spin2', 'predicted_snr', 'TPE'])
    GWtunaParamsRecovery.to_csv(OUTPUT_FILE2, index = False)

