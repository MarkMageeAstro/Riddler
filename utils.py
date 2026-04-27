
import re
import os
import glob

from pickle import load

import ultranest
import ultranest.stepsampler

import numpy as np
import pandas as pd
import scipy
from astropy import units as u
from astropy import constants as csts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from extinction import ccm89, fitzpatrick99, apply


def run_ultranest_fitting(root_dir, sn_name, model_type, nn_filename, restart_flag, prediction_handler):
    """
    Function to run the UltraNest fitting procedure

    Parameters
    ----------
    root_dir : str
        The root directory where the data and outputs are stored.
    sn_name : str
        The name of the supernova being fitted.
    model_type : str
        The type of explosion scenario to consider when generating the spectra. Currently this is one of DEF, DDT, DOD, GCD, or VM.
    nn_filename : str
        The filename of the trained neural network to use for generating the predicted spectra during fitting.
    restart_flag : bool
        A flag to indicate whether to restart the fitting procedure from a previous run. If True, the fitting will be restarted from the last saved state. If False, the fitting will start from scratch.
    """

    ## Set the output path where the results will be saved
    output_path = root_dir + "Outputs/" + sn_name + "/" + "FittedType_" + model_type + "/" + nn_filename
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    ## If the restart flag is set to True, find the last saved state and resume the fitting procedure from there. Otherwise, start from scratch
    if restart_flag:
        print("Resuming fitting procedure from previous run.", flush=True)
        resume_flag = 'resume'
        subfolders = sorted(glob.glob(output_path + "/run*"), key=extract_run_number)
        output_path = subfolders[-1]
    else:
        print("Starting fitting procedure from scratch.", flush=True)
        resume_flag = 'subfolder'    
        

    ## Set up the UltraNest sampler
    sampler = ultranest.ReactiveNestedSampler(prediction_handler.parameter_names, prediction_handler.log_likelihood, prediction_handler.prior_transform, vectorized=True, log_dir=output_path, resume=resume_flag)

    ## Run the sampler initially without the step sampler, this is designed to make it a bit faster initially
    sampler.run(show_status=True, min_num_live_points=4000, max_ncalls=2e4)

    ## After the initial run, switch to using the step sampler to make it more efficient at sampling the parameter space
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=20, generate_direction=ultranest.stepsampler.generate_mixture_random_direction,)
    sampler.run(show_status=True, min_num_live_points=4000, max_ncalls=2e4)

    ## Print and plot the results
    sampler.print_results()

    try:
        sampler.plot_run()
    except Exception as e:
        print("Could not generate run plot: %s", e, flush=True)

    try:
        sampler.plot_trace()
    except Exception as e:
        print("Could not generate trace plot: %s", e, flush=True)

    return sampler



def plot_spectrum(root_dir, sn_name, data_handler, model_type, nn_filename, input_parameters, prediction_handler):
    """
    Function to plot the predicted spectrum for a given set of input parameters.
    Not vectorized, so only designed to plot a single set of spectra

    Parameters
    ----------
    root_dir : str
        The root directory where the data and outputs are stored.
    sn_name : str
        The name of the supernova being fitted.
    model_type : str
        The type of explosion scenario to consider when generating the spectra. Currently this is one of DEF, DDT, DOD, GCD, or VM.
    nn_filename : str
        The filename of the trained neural network to use for generating the predicted spectra during fitting.
    input_parameters : array-like : float
        The set of input parameters to use for generating the predicted spectrum.

    """

    ## Set path where results are stored
    ## Will run on the most recent fit
    output_path = root_dir + "Outputs/" + sn_name + "/" + "FittedType_" + model_type + "/" + nn_filename
    subfolders = sorted(glob.glob(output_path + "/run*"), key=extract_run_number)
    output_path = subfolders[-1]

    ## Get the spectra and errrors
    predicted_flux, predicted_flux_std = prediction_handler.predict_spectra(input_parameters)

    # Generate plot of predicted spectra vs. observed spectra
    plt.figure(num=None, figsize=(6, len(data_handler.observed_fluxes) * 2.5), dpi=300, facecolor='w', edgecolor='k')
    ## Make a separate panel for each spectrum
    grid = plt.GridSpec(len(data_handler.observed_spectra_dates), 1, wspace=0.2, hspace=0)
    ax = []
    ## For each phase, plot the observed and predicted spectra. Shaded regions denote the 1sigma error range
    for i in np.arange(len(data_handler.observed_fluxes)):

        ax.append(plt.subplot(grid[i]))

        ax[i].minorticks_on()
        ax[i].tick_params(axis="both", which="major", direction="in", top='On', right='On', length=8)
        ax[i].tick_params(axis="both", which="minor", direction="in", top='On', right='On', length=4)

        ax[i].set_xlim(2005, 10000)
        
        ax[i].fill_between(data_handler.wavelengths, data_handler.observed_fluxes[i] - data_handler.observed_flux_errors[i], data_handler.observed_fluxes[i] + data_handler.observed_flux_errors[i], color='k', alpha=0.5, edgecolor='None')
        ax[i].plot(data_handler.wavelengths, data_handler.observed_fluxes[i], color='k', label='Observed Spectrum', linewidth=2)

        ax[i].fill_between(data_handler.wavelengths, predicted_flux[i] - predicted_flux_std[i], predicted_flux[i] + predicted_flux_std[i], color='r', alpha=0.5, edgecolor='None')
        ax[i].plot(data_handler.wavelengths, predicted_flux[i], color='r', linewidth=2, label='NN prediction')
        
        if i < (len(data_handler.observed_fluxes) - 1):
            ax[i].set_xticklabels([])
            
        ax[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax[i].set_ylabel("Observed flux")

        
    ax[-1].legend()
    ax[-1].set_xlabel("Rest wavelength")

    ## Save the plot to the UltraNest plots directory
    plt.savefig(output_path + '/plots/best_fit_predictions.pdf', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

    return None



def process_posteriors(root_dir, sn_name, nn_filename, results, data_handler, model_type):

    """
    Function to process the posterior samples from UltraNest and calculate physically meaningful derived properties not directly sampled, such as Ni mass, luminosity, etc.
    """

    redshift_factor = (1. + data_handler.observed_properties['Redshift'])

    ## Go through each spectrum and calculate the phase and inner boundary velocity
    for i in np.arange(len(data_handler.observed_spectra_dates)):
        results['phase_spec' + str(i+1)] = (data_handler.observed_spectra_dates[i] - results['time_explosion_jd']) / redshift_factor
        results['v_inner_spec' + str(i+1)] = results['velocity_at_peak_spec' + str(i+1)] - results['velocity_gradient_spec' + str(i+1)] * ( results['phase_spec' + str(i+1)] - results['rise_time'] )

    ## The following masses are all just calculated as products
    results['core_burned_mass']     = results['total_ejecta_mass'] * results['core_fraction']       * results['core_fraction_burned']
    results['core_ige_mass']        = results['total_ejecta_mass'] * results['core_fraction']       * results['core_fraction_burned']       * results['core_fraction_ige/burned']
    results['core_ni56_mass']       = results['total_ejecta_mass'] * results['core_fraction']       * results['core_fraction_burned']       * results['core_fraction_ige/burned']           * results['core_fraction_ni56/ige']
    results['core_ime_mass']        = results['total_ejecta_mass'] * results['core_fraction']       * results['core_fraction_burned']       * (1 - results['core_fraction_ige/burned'])
    results['core_unburned_mass']   = results['total_ejecta_mass'] * results['core_fraction']       * (1 - results['core_fraction_burned'])
    results['shell_burned_mass']    = results['total_ejecta_mass'] * (1 - results['core_fraction']) * results['shell_fraction_burned']
    results['shell_ige_mass']       = results['total_ejecta_mass'] * (1 - results['core_fraction']) * results['shell_fraction_burned']      * results['shell_fraction_ige/burned']
    results['shell_ni56_mass']      = results['total_ejecta_mass'] * (1 - results['core_fraction']) * results['shell_fraction_burned']      * results['shell_fraction_ige/burned']          * results['shell_fraction_ni56/ige']
    results['shell_ime_mass']       = results['total_ejecta_mass'] * (1 - results['core_fraction']) * results['shell_fraction_burned']      * (1 -results ['shell_fraction_ige/burned'])
    results['shell_unburned_mass']  = results['total_ejecta_mass'] * (1 - results['core_fraction']) * (1-results['shell_fraction_burned'])

    ## Calculate the luminosity at the phase of the observed spectrum
    ## The lumionsity depends on the model type and underlying models used to construct the training data, so this is calculated separately for each model type
    ## Load up the luminosity interpolator for the relevant model type
    if model_type == "deflagration":
        training_path   = root_dir + "Inputs/TrainingData/Deflagrations/Fink_2014/"
    elif model_type == "delayed_detonation":
        training_path   = root_dir + "Inputs/TrainingData/DelayedDetonations/Seitenzahl_2013/"
    elif model_type == "double_detonation":
        training_path   = root_dir + "Inputs/TrainingData/DoubleDetonations/Gronow_2021/"
    elif model_type == "gravitationally_confined_detonation":
        training_path   = root_dir + "Inputs/TrainingData/GravitationallyConfinedDetonations/Lach_2021/"
    elif model_type == "mergers":
        training_path   = root_dir + "Inputs/TrainingData/Mergers/"

    luminosity_interpolator = load(open(training_path + "Interpolators/Luminosity_Interpolator.pkl", 'rb'))

    orig_times = np.arange(5, 35, 0.01)
    if model_type == 'deflagration':
        luminosity = np.array([f(np.log10(results['independent'].values)) for f in luminosity_interpolator])
        luminosity = luminosity / np.max(luminosity, axis=0) * np.power(10, results['log_peak_flux'].values)
        orig_rise_time = orig_times[np.argmax(luminosity, axis=0)]
        orig_times_array = np.repeat(orig_times[np.newaxis,:], len(results['log_peak_flux'].values), axis=0)
        scaled_times = (orig_times_array - orig_rise_time[:, np.newaxis]) * (results['rise_time'].values / orig_rise_time)[:,np.newaxis] + results['rise_time'].values[:, np.newaxis]
        luminosity_list = []
        for row in np.arange(len(results)):
            luminosity_spec_list = []
            for i in np.arange(len(data_handler.observed_spectra_dates)):
                luminosity_spec = np.interp(results.loc[row, 'phase_spec' + str(i+1)], scaled_times[row,:], luminosity[:,row])
                luminosity_spec_list.append(luminosity_spec)
            luminosity_list.append(np.log10(luminosity_spec_list / csts.L_sun.to('erg/s').value))
    elif model_type == 'delayed_detonation':
        luminosity = np.array([f(np.log10(results['independent'].values)) for f in luminosity_interpolator])
        luminosity = luminosity / np.max(luminosity, axis=0) * np.power(10, results['log_peak_flux'].values)
        orig_rise_time = orig_times[np.argmax(luminosity, axis=0)]
        orig_times_array = np.repeat(orig_times[np.newaxis,:], len(results['log_peak_flux'].values), axis=0)
        scaled_times = (orig_times_array - orig_rise_time[:, np.newaxis]) * (results['rise_time'].values / orig_rise_time)[:,np.newaxis] + results['rise_time'].values[:, np.newaxis]
        luminosity_list = []
        for row in np.arange(len(results)):
            luminosity_spec_list = []
            for i in np.arange(len(data_handler.observed_spectra_dates)):
                luminosity_spec = np.interp(results.loc[row, 'phase_spec' + str(i+1)], scaled_times[row,:], luminosity[:,row])
                luminosity_spec_list.append(luminosity_spec)
            luminosity_list.append(np.log10(luminosity_spec_list / csts.L_sun.to('erg/s').value))
    elif model_type == 'double_detonation':
        luminosity = np.array([f(results['total_ejecta_mass'].values) for f in luminosity_interpolator])
        luminosity = luminosity / np.max(luminosity, axis=0) * np.power(10, results['log_peak_flux'].values)
        orig_rise_time = orig_times[np.argmax(luminosity, axis=0)]
        orig_times_array = np.repeat(orig_times[np.newaxis,:], len(results['log_peak_flux'].values), axis=0)
        scaled_times = (orig_times_array - orig_rise_time[:, np.newaxis]) * (results['rise_time'].values / orig_rise_time)[:,np.newaxis] + results['rise_time'].values[:, np.newaxis]
        luminosity_list = []
        for row in np.arange(len(results)):
            luminosity_spec_list = []
            for i in np.arange(len(data_handler.observed_spectra_dates)):
                luminosity_spec = np.interp(results.loc[row, 'phase_spec' + str(i+1)], scaled_times[row,:], luminosity[:,row])
                luminosity_spec_list.append(luminosity_spec)
            luminosity_list.append(np.log10(luminosity_spec_list / csts.L_sun.to('erg/s').value))
    elif model_type == 'gravitationally_confined_detonation':
        luminosity = np.array([f(np.log10(results['independent'].values)) for f in luminosity_interpolator])
        luminosity = luminosity / np.max(luminosity, axis=0) * np.power(10, results['log_peak_flux'].values)
        orig_rise_time = orig_times[np.argmax(luminosity, axis=0)]
        orig_times_array = np.repeat(orig_times[np.newaxis,:], len(results['log_peak_flux'].values), axis=0)
        scaled_times = (orig_times_array - orig_rise_time[:, np.newaxis]) * (results['rise_time'].values / orig_rise_time)[:,np.newaxis] + results['rise_time'].values[:, np.newaxis]
        luminosity_list = []
        for row in np.arange(len(results)):
            luminosity_spec_list = []
            for i in np.arange(len(data_handler.observed_spectra_dates)):
                luminosity_spec = np.interp(results.loc[row, 'phase_spec' + str(i+1)], scaled_times[row,:], luminosity[:,row])
                luminosity_spec_list.append(luminosity_spec)
            luminosity_list.append(np.log10(luminosity_spec_list / csts.L_sun.to('erg/s').value))
    elif model_type == 'mergers':
        luminosity = np.array([f(results['total_ejecta_mass'].values) for f in luminosity_interpolator])
        luminosity = luminosity / np.max(luminosity, axis=0) * np.power(10, results['log_peak_flux'].values)
        orig_rise_time = orig_times[np.argmax(luminosity, axis=0)]
        orig_times_array = np.repeat(orig_times[np.newaxis,:], len(results['log_peak_flux'].values), axis=0)
        scaled_times = (orig_times_array - orig_rise_time[:, np.newaxis]) * (results['rise_time'].values / orig_rise_time)[:,np.newaxis] + results['rise_time'].values[:, np.newaxis]
        luminosity_list = []
        for row in np.arange(len(results)):
            luminosity_spec_list = []
            for i in np.arange(len(data_handler.observed_spectra_dates)):
                luminosity_spec = np.interp(results.loc[row, 'phase_spec' + str(i+1)], scaled_times[row,:], luminosity[:,row])
                luminosity_spec_list.append(luminosity_spec)
            luminosity_list.append(np.log10(luminosity_spec_list / csts.L_sun.to('erg/s').value))
    

    ## Save the processed posteriors to a .csv file in the same directory as the UltraNest results
    output_path = root_dir + "Outputs/" + sn_name + "/FittedType_" + model_type + "/" + nn_filename
    subfolders = sorted(glob.glob(output_path + "/run*"), key=extract_run_number)
    output_path = subfolders[-1]
    results.to_csv(output_path + "/chains/equal_weighted_post_processed.txt")


def extract_number(filename):
    """
    Function to make sure files are sorted numerically
    """
    match = re.search(r'param(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # return 0 if no match


def get_interpolators(root_dir, model_type):
    """
    Function to get the interpolators that are used for parameter sampling
    """
    
    ## Get the list of interpolator files for the relevant model type
    if model_type == "deflagration":
        training_path   = root_dir + "Inputs/TrainingData/Deflagrations/Fink_2014/"
    elif model_type == "delayed_detonation":
        training_path   = root_dir + "Inputs/TrainingData/DelayedDetonations/Seitenzahl_2013/"
    elif model_type == "double_detonation":
        training_path   = root_dir + "Inputs/TrainingData/DoubleDetonations/Gronow_2021/"
    elif model_type == "gravitationally_confined_detonation":
        training_path   = root_dir + "Inputs/TrainingData/GravitationallyConfinedDetonations/Lach_2021/"
    elif model_type == "mergers":
        training_path   = root_dir + "Inputs/TrainingData/Mergers/"

    parameter_limit_files = sorted(glob.glob(training_path + "Limits/param*"), key=extract_number)

    ## Create a list to hold all of the interpolators
    interpolator_list = []
    ## For each parameter file, read the file, and create an interpolators that will be used to set minimum and maximum boundaries for sampling
    for parameter_file in parameter_limit_files:

        limits = pd.read_csv(parameter_file)
        limits = limits.loc[limits.iloc[:,0] > -np.inf]

        ## There are different behaviours for the limits
        ## Some values are just fixed in some cases, e.g. DDT models all have 1.4 solar masses of ejecta
        ## In some cases it's just uniform sampling between certain values, either using X or log X to determine the appropriate boundaries
        ## In other cases, it could be arbitrary 
        if 'fixed' in parameter_file:
            interpolator_lower_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['fixed'].iloc[0], limits['fixed'].iloc[0]], k=1)
            interpolator_upper_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['fixed'].iloc[0], limits['fixed'].iloc[0]], k=1)
        elif 'uniform' in parameter_file:
            if 'log' in limits.columns[0]:
                interpolator_lower_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['log_lower_limit'].iloc[0], limits['log_lower_limit'].iloc[0]], k=1)
                interpolator_upper_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['log_upper_limit'].iloc[0], limits['log_upper_limit'].iloc[0]], k=1)
            else:
                interpolator_lower_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['lower_limit'].iloc[0], limits['lower_limit'].iloc[0]], k=1)
                interpolator_upper_limit = scipy.interpolate.make_interp_spline([-1e100, 1e100], [limits['upper_limit'].iloc[0], limits['upper_limit'].iloc[0]], k=1)
        else:
            lower_limit_column_name = limits.columns[limits.columns.str.contains("lower")]
            upper_limit_column_name = limits.columns[limits.columns.str.contains("upper")]

            interpolator_lower_limit = scipy.interpolate.make_interp_spline(limits.iloc[:,0].values, limits[lower_limit_column_name].iloc[:,0].values, k=1)
            interpolator_upper_limit = scipy.interpolate.make_interp_spline(limits.iloc[:,0].values, limits[upper_limit_column_name].iloc[:,0].values, k=1)

        interpolator_list.append( [interpolator_lower_limit, interpolator_upper_limit] )

    return interpolator_list



def extract_run_number(path):
    match = re.search(r'run(\d+)', path)
    return int(match.group(1)) if match else -1
