import sys
import glob
from pickle import load

import warnings
warnings.filterwarnings("ignore", message='', category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", message='', category=UserWarning, module='ultranest')

import numpy as np
import pandas as pd

import torch
import torch.optim as optim


import utils
from nn_models import NeuralNetwork
from data import ObservedData
from predict_spectra import SpectralPredictions



root_dir = './'
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"



## Arguments
## Name of the SN to be fit
sn_name = sys.argv[1]

## Which neural network model will be used during fitting
## Currently one of deflagration (DEF), delayed detonation (DDT), double detonation (DOD), gravitationally confined detonation (GCD), or violent merger (VM)
model_type = sys.argv[2]

## Flag for whether to continue a previous UltraNest run or start a new one
restart_flag = sys.argv[3].lower() == 'true'

## Define the wavelength grid of the neural network
## Currently only the values shown here are supported, but future neural networks could be trained over different wavelength grids
START = 2000
STOP = 10000
BINS = 2000
SPACING_TYPE = 'log'
SCALE_TYPE = 'standard'
if SPACING_TYPE == 'log':
    wavelengths = np.logspace(np.log10(START), np.log10(STOP), BINS )
elif SPACING_TYPE == 'linear':
    wavelengths = np.linspace(START, STOP, BINS )


## Load up neural network
## Currently only a single set of hyperparameters is supported, the same for all models, but future neural networks could have different parameters
if model_type == "DEF":
    model_type      = 'deflagration'
    nn_path         = root_dir + "NeuralNetworks/deflagration/P2S_NN/start-" + str(START) + "_stop-" + str(STOP) + "_bins-" + str(BINS) + "_spacing-" + SPACING_TYPE + "/scaler-" + SCALE_TYPE + "/NN_"
    training_path   = root_dir + "Inputs/TrainingData/Deflagrations/Fink_2014/"
elif model_type == "DDT":
    model_type      = 'delayed_detonation'
    nn_path         = root_dir + "NeuralNetworks/delayed_detonation/P2S_NN/start-" + str(START) + "_stop-" + str(STOP) + "_bins-" + str(BINS) + "_spacing-" + SPACING_TYPE + "/scaler-" + SCALE_TYPE + "/NN_"
    training_path   = root_dir + "Inputs/TrainingData/DelayedDetonations/Seitenzahl_2013/"
elif model_type == "DOD":
    model_type      = 'double_detonation'
    nn_path         = root_dir + "NeuralNetworks/double_detonation/P2S_NN/start-" + str(START) + "_stop-" + str(STOP) + "_bins-" + str(BINS) + "_spacing-" + SPACING_TYPE + "/scaler-" + SCALE_TYPE + "/NN_"
    training_path   = root_dir + "Inputs/TrainingData/DoubleDetonations/Gronow_2021/"
elif model_type == "GCD":
    model_type      = 'gravitationally_confined_detonation'
    nn_path         = root_dir + "NeuralNetworks/gravitationally_confined_detonation/P2S_NN/start-" + str(START) + "_stop-" + str(STOP) + "_bins-" + str(BINS) + "_spacing-" + SPACING_TYPE + "/scaler-" + SCALE_TYPE + "/NN_"
    training_path   = root_dir + "Inputs/TrainingData/GravitationallyConfinedDetonations/Lach_2021/"
elif model_type == "VM":
    model_type      = 'mergers'
    nn_path         = root_dir + "NeuralNetworks/mergers/P2S_NN/start-" + str(START) + "_stop-" + str(STOP) + "_bins-" + str(BINS) + "_spacing-" + SPACING_TYPE + "/scaler-" + SCALE_TYPE + "/NN_"
    training_path   = root_dir + "Inputs/TrainingData/Mergers/"

## Define the parameters used to train the neural network
warm_up = 1000
batch_size = 8192
learning_rate = 1.0e-03
layers = 4
init_sigma = -4.
neurons = 400
KL = 1.00e-04
activation = 'Softplus'
optimizer = 'AdamW'

input_shape = 16
output_shape = BINS

nn_filename = "warm-" + str(warm_up) + "_bs-" + str(batch_size) + "_lr-" + "{:.2e}".format(learning_rate) + "_layers-" + str(layers) + "_neurons-" + str(neurons) + "_activation-" + activation + "_optimizer-" + optimizer

## Load the trained model
loaded_model = NeuralNetwork(input_shape, output_shape, neurons, layers, activation).to(device)
loaded_optimizer = getattr(optim, optimizer)
loaded_optimizer = loaded_optimizer(loaded_model.parameters(), lr=learning_rate)

checkpoint = torch.load(nn_path + nn_filename + "/checkpoint_best_model.pth", map_location=device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

## Set to evaluation mode
loaded_model.eval()


## Load up the scalers that were used to standardise the training data and labels
with open(training_path + model_type + "_binned_start_wave" + str(START) + "_stop_wavelength" + str(STOP) + "_bins" + str(BINS) + "_" + SPACING_TYPE + "_properties_standard_scale_fit_trimmed.pkl", 'rb') as label_fit_file:
    physical_properties_scaler = load(label_fit_file)

with open(training_path + model_type + "_binned_start_wave" + str(START) + "_stop_wavelength" + str(STOP) + "_bins" + str(BINS) + "_" + SPACING_TYPE + "_spectra_standard_scale_fit_trimmed.pkl", 'rb') as data_fit_file:
    physical_flux_scaler = load(data_fit_file)

## Load up the scale factors for the uncertainty
## This is a wavelength dependent array of factors by which to increase the predicted uncertainty from the neural network
## Its purpose is to ensure that predicted uncertainties have not been underestimated
uncertainty_scale_factor = pd.read_csv(nn_path + nn_filename + "/predicted_uncertainty_scale_factors.csv")
uncertainty_scale_factor = uncertainty_scale_factor['Scale_Factor_50'].values



## Load up the list of interpolators that will be used during parameter sampling
interpolator_list = utils.get_interpolators(root_dir, model_type)



## Create a data handler that will read and store the observed data and properties
data_handler = ObservedData(root_dir, sn_name, wavelengths)


## Define the parameter names to be used
## log_f is a nuisance parameter to account for systematic differences between the models and the data
## time_explosion_jd, distance_modulus, host_ebv, host_rv are parameters related to the observations and apply to all spectra
## The rest are the physical parameters of the explosion model and were used in the training of the neural network. These also apply to all spectra
param_names = ['log_f', 
               'time_explosion_jd', 'distance_modulus', 'host_ebv', 'host_rv', 
               'independent', 
               'total_ejecta_mass', 'core_fraction', 'core_fraction_burned', 'core_fraction_ige/burned', 'core_fraction_ni56/ige', 'shell_fraction_burned', 'shell_fraction_ige/burned', 'shell_fraction_ni56/ige', 'shell_fraction_unburned_co', 
               'log_KE', 'rise_time', 'log_peak_flux'
               ]
## For each spectrum, we also have velocity parameters that determine the inner boundary velocity
for i in np.arange(len(data_handler.observed_spectra_dates)):
    param_names.append('velocity_at_peak_spec' + str(i+1))
    param_names.append('velocity_gradient_spec' + str(i+1))

nsteps = len(param_names)




## Main function
def main():
    print("Running RIDDLER", flush=True)

    ## Create the prediction handler
    ## This class is responsible for sampling for transforming from the prior, generating the spectra, and calculating the likelihood against the data (handler)
    prediction_handler = SpectralPredictions(data_handler, model_type, physical_properties_scaler, physical_flux_scaler, interpolator_list, device, loaded_model, param_names, uncertainty_scale_factor)

    ## Run the fitting function
    utils.run_ultranest_fitting(root_dir, sn_name, model_type, nn_filename, restart_flag, prediction_handler)
    print("Finished RIDDLER run", flush=True)

    ## Load up the results from the run
    ## Take the mean of the posteriors, then plot and process for quick comparisons later
    output_path = root_dir + "Outputs/" + sn_name + "/FittedType_" + model_type + "/" + nn_filename
    subfolders = sorted(glob.glob(output_path + "/run*"), key=utils.extract_run_number)
    output_path = subfolders[-1]
    print("Loading results from: " + output_path, flush=True)

    equal_weighted_post = pd.read_csv(output_path + "/chains/equal_weighted_post.txt", sep='\s+')
    best_fitting_parameters = equal_weighted_post.mean().values

    print("Plotting best fitting parameters", flush=True)
    utils.plot_spectrum(root_dir, sn_name, data_handler, model_type, nn_filename, best_fitting_parameters, prediction_handler)

    print("Processing posterior distributions", flush=True)
    utils.process_posteriors(root_dir, sn_name, nn_filename, equal_weighted_post, data_handler, model_type)

    print("Finished!", flush=True)



if __name__ == "__main__":
    main()
