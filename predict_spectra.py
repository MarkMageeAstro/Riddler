import torch

import scipy
import numpy as np
from astropy import units as u

from extinction import ccm89, fitzpatrick99, apply


class SpectralPredictions:
    """
    Class to handle generating the neural network predictions and calculating the likelihood against the observed spectra.
    """
    def __init__(self, data_handler, model_type, physical_properties_scaler, physical_flux_scaler, interpolator_list, device, trained_nn, parameter_names, uncertainty_scale_factor):
        """ 
        Initializes predictions class.

        Parameters
        ----------
        observed_properties : dataframe
            A dataframe containing some properties of the observed supernova that are used during fitting, including redshift, explosion time range, etc.
        observed_spectra_dates : array-like : float
            The dates at which the real spectra were observed.
        observed_weights : array-like : float
            An array of wavelength-dependent weights to apply to the likelihood calculation.
        observed_errors : array-like : float
            The errors on the observed fluxes for each wavelength bin and each spectrum.
        observed_fluxes : array-like : float
            The observed fluxes for each wavelength bin and each spectrum.
        model_type : str
            The type of explosion scenario to consider when generating the spectra. Currently this is one of DEF, DDT, DOD, GCD, or VM.
        physical_properties_scaler : sklearn.preprocessing.StandardScaler
            The scaler used to scale the physical properties of the training data so that they could be used to train the neural network. Used here to convert to/from scaled values during prediction.
        physical_flux_scaler : sklearn.preprocessing.StandardScaler
            The scaler used to scale the flux values of the spectral training data so that they could be used to train the neural network. Used here to convert to/from scaled values during prediction.
        interpolator_list : list of scipy.interpolate.interp1d objects
            A list of interpolators from which to sample parameters based on previously generated values, e.g. to sample the ejecta mass based on the number of sparks for deflagrations.
        device : torch.device
            The device to use for the neural network predictions (e.g. 'cpu' or 'cuda').
        trained_nn : torch.nn.Module
            The trained neural network model used to generate the predictions.
        parameter_names : list of str
            The names of the input parameters that will be fit by UltraNest
        uncertainty_scale_factor : array-like : float
            An array of wavelength-dependent scale factors to apply to the predicted uncertainties to account for any underestimation of the uncertainties by the neural network. 
        wavelengths : array-like : float
            The wavelength grid of the neural network.
        """

        ## Observational parameters
        self.observed_properties = data_handler.observed_properties
        self.observed_spectra_dates = data_handler.observed_spectra_dates
        ## Observed data
        self.observed_fluxes = data_handler.observed_fluxes
        self.observed_flux_errors = data_handler.observed_flux_errors
        self.observed_flux_weights = data_handler.observed_flux_weights
        ## The earliest/latest explosion date to consider during fitting. Should match the units of observed_spectra_dates, e.g. JD, MJD, etc.
        self.t_earliest = data_handler.observed_properties['t_exp_min']
        self.t_latest = data_handler.observed_properties['t_exp_max']
        ## The redshift factor to apply for the conversion between observer and rest frame.
        self.redshift_factor = (1. + data_handler.observed_properties['Redshift'])

        ## Neural network parameters
        self.model_type = model_type
        self.physical_properties_scaler = physical_properties_scaler
        self.physical_flux_scaler = physical_flux_scaler
        self.interpolator_list = interpolator_list
        self.device = device
        self.trained_nn = trained_nn
        self.parameter_names = parameter_names
        self.uncertainty_scale_factor = uncertainty_scale_factor
        self.wavelengths = data_handler.wavelengths
        self.n_bins = len(self.wavelengths)


    def prior_transform(self, cube):
        """
        Function to transform the input parameters from the unit cube to the physical parameters used by the neural network.
        Some additional parameters are generated that are not directly used by the neural network, but required to properly sample the input parameter space.
        """
        
        params = cube.copy()

        ## Transform log_f
        ## log_f is assumed to range from ~8% up to 100%
        params[:, 0] = cube[:, 0] * (0. - -2.5) + -2.5

        ## Transform time of explosion
        ## First spectrum must be at least 5d after explosion
        ## Last spectrum must be at most 30d after explosion
        ## Asumes a uniform distribution between boundaries set by the user, but could be changed to a normal distibution or something else if desired
        params[:, 1] = cube[:, 1] * ( self.t_latest - self.t_earliest ) + self.t_earliest

        ## Transform distance modulus
        ## Distance modulus is assumed to be given by a Gaussian distribution with mean and std given by the user
        ## Otherwise if no error is given then just fix the distance modulus to the given value
        if self.observed_properties['Distance_modulus_err'] > 0:
            params[:, 2] = scipy.stats.norm.ppf(cube[:, 2], loc=self.observed_properties['Distance_modulus'], scale=self.observed_properties['Distance_modulus_err'])
        else:
            params[:, 2] = self.observed_properties['Distance_modulus']

        ## Transform host galaxy extinction parameters
        ## E(B-V) is assumed to be given by a Gaussian distribution with mean and std given by the user
        ## If no error is given, but E(B-V) > 0, then fix the value to that given by the user
        ## If no E(B-V) is given either, then can fit for it assuming an exponential distribution with a scale of 0.11
        ## Rv is allowed to vary and also assumed to be given by a Gaussian distribution with mean and std given by the user
        ## If no error is given, but Rv > 0, then fix the value to that given by the user
        ## If no Rv is given either, then just use a fixed Rv = 3.1
        if self.observed_properties['E(B-V)_host_err'] > 0.:
            params[:, 3] = scipy.stats.norm.ppf(cube[:, 3], loc=self.observed_properties['E(B-V)_host'], scale=self.observed_properties['E(B-V)_host_err'])
        elif self.observed_properties['E(B-V)_host'] > 0.:
            params[:, 3] = self.observed_properties['E(B-V)_host']
        elif self.observed_properties['E(B-V)_host'] == -999:
            params[:, 3] = scipy.stats.expon.ppf(cube[:, 3], scale=0.11)
        else:
            params[:, 3] = 0.0

        if self.observed_properties['Rv_host_err'] > 0.:
            params[:, 4] = scipy.stats.norm.ppf(cube[:, 4], loc=self.observed_properties['Rv_host'], scale=self.observed_properties['Rv_host_err'])
        elif self.observed_properties['Rv_host'] > 0.:
            params[:, 4] = self.observed_properties['Rv_host']
        else:
            params[:, 4] = 3.1

        
        ## The remaining parameters are all related to the type of explosion
        ## The input parameters themselves are the same for all types of explosion scenarios, but were generated in different ways therefore different methods are required
        if self.model_type == 'deflagration':
            ## ***************************
            ## Ejecta structure parameters
            ## ***************************
            ## Number of sparks is the independent parameter from which many other parameters are derived
            params[:,5] = cube[:,5] * ( 20. - 1. ) + 1.
            log_number_of_sparks = np.log10(params[:,5])

            ## Total ejecta mass depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[1][1](log_number_of_sparks)
            interpolator_min_boundary = self.interpolator_list[1][0](log_number_of_sparks)
            log_ejecta_mass = cube[:,6] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary
            params[:,6] = np.power(10, log_ejecta_mass )
        
            ## Core fraction is fixed to 1. for deflagrations
            params[:,7] = 1.

            ## Core fraction burned depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[3][1](log_number_of_sparks)
            interpolator_min_boundary = self.interpolator_list[3][0](log_number_of_sparks)
            params[:,8] = cube[:,8] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ige/burned depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[4][1](log_number_of_sparks)
            interpolator_min_boundary = self.interpolator_list[4][0](log_number_of_sparks)
            params[:,9] = cube[:,9] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ni56/ige depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[5][1](log_number_of_sparks)
            interpolator_min_boundary = self.interpolator_list[5][0](log_number_of_sparks)
            params[:,10] = cube[:,10] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell parameters are all 0 for deflagrations
            params[:,11] = 0.
            params[:,12] = 0.
            params[:,13] = 0.
            params[:,14] = 0.

            ## Kinetic energy depends on ejecta mass
            interpolator_max_boundary = self.interpolator_list[10][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[10][0](log_ejecta_mass)
            params[:,15] = cube[:,15] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## ***************************
            ## Simulation input parameters
            ## ***************************
            ## This includes the luminosity and inner boundary velocity for each spectrum
            ## To determine the luminosity, first need to determine a light curve and from that the luminosity at each epoch
            ## The light curve is based on the underlying explosion simulations and scaled to a new rise time and a new peak flux
        
            ## Rise time depends on the ejecta mass
            interpolator_max_boundary = self.interpolator_list[11][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[11][0](log_ejecta_mass)
            params[:,16] = cube[:,16] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Peak flux depends on the 56Ni mass
            ni56_mass = params[:,6] * params[:,7] * params[:,8] * params[:,9] * params[:,10]
            log_ni56_mass = np.log10(ni56_mass)
            interpolator_max_boundary = self.interpolator_list[12][1](log_ni56_mass)
            interpolator_min_boundary = self.interpolator_list[12][0](log_ni56_mass)
            params[:,17] = cube[:,17] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## The inner boundary velocity is given by the time, velocity at peak, and velocity gradient for each spectrum
            ## For the first spectrum, the velocity at peak is sampled based on the number of sparks, and the velocity gradient is sampled independently
            ## For subsequent spectra, the velocity gradient is sample independently, but the velocity at peak is constrained based on the previous spectrum to ensure that the inner boundary velocity decreases over time
            interpolator_max_boundary = self.interpolator_list[13][1](log_number_of_sparks)
            interpolator_min_boundary = self.interpolator_list[13][0](log_number_of_sparks)
            params[:,18] = cube[:,18] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary
            params[:,19] = cube[:,19] * ( 250. - 50. ) + 50.
            phases = (self.observed_spectra_dates - params[:,1][:, None]) / self.redshift_factor

            for i in np.arange(1, len(self.observed_spectra_dates)):
                previous_phase = phases[:,i-1]
                phase = phases[:,i]

                params[:, 19 + i*2] = cube[:, 19 + i*2] * ( 250. - 50. ) + 50.

                max_allowed_velocity_at_peak = params[:,18 + (i-1)*2] - params[:,19 + (i-1)*2] * ( previous_phase - params[:,16] ) + params[:, 19 + i*2] * ( phase - params[:,16] )
                max_allowed_velocity_at_peak[max_allowed_velocity_at_peak > interpolator_max_boundary] = interpolator_max_boundary[max_allowed_velocity_at_peak > interpolator_max_boundary]

                min_allowed_velocity_at_peak = 4000. + params[:, 19 + i*2] * ( phase - params[:,16] )
                min_allowed_velocity_at_peak[min_allowed_velocity_at_peak < interpolator_min_boundary] = interpolator_min_boundary[min_allowed_velocity_at_peak < interpolator_min_boundary]

                params[:,18 + i*2] = cube[:,18 + i*2] * ( max_allowed_velocity_at_peak - min_allowed_velocity_at_peak ) + min_allowed_velocity_at_peak
        elif self.model_type == 'delayed_detonation':
            ## ***************************
            ## Ejecta structure parameters
            ## ***************************
            ## Number of sparks is the independent parameter from which many other parameters are derived
            params[:,5] = cube[:,5] * ( 150. - 10. ) + 10.

            ## Total ejecta mass is fixed
            params[:,6] = 1.400486

            ## Core fraction is fixed to 1. for delayed detonations
            params[:,7] = 1.

            ## Core fraction burned depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[3][1](params[:,5])
            interpolator_min_boundary = self.interpolator_list[3][0](params[:,5])
            params[:,8] = cube[:,8] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ige/burned depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[4][1](params[:,5])
            interpolator_min_boundary = self.interpolator_list[4][0](params[:,5])            
            params[:,9] = cube[:,9] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ni56/ige depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[5][1](params[:,5])
            interpolator_min_boundary = self.interpolator_list[5][0](params[:,5])  
            params[:,10] = cube[:,10] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell parameters are all 0 for delayed detonations
            params[:,11] = 0.
            params[:,12] = 0.
            params[:,13] = 0.
            params[:,14] = 0.

            ## Kinetic energy depends on number of sparks
            interpolator_max_boundary = self.interpolator_list[10][1](params[:,5])
            interpolator_min_boundary = self.interpolator_list[10][0](params[:,5])  
            params[:,15] = np.log10(cube[:,15] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary)

            ## ***************************
            ## Simulation input parameters
            ## ***************************
            ## This includes the luminosity and inner boundary velocity for each spectrum
            ## To determine the luminosity, first need to determine a light curve and from that the luminosity at each epoch
            ## The light curve is based on the underlying explosion simulations and scaled to a new rise time and a new peak flux

            ## Rise time technically depends on the number of sparks, but in practice shows little variation therefore just uniformly sample
            params[:,16] = cube[:,16] * ( 22.68 - 15.73 ) + 15.73

            ## Peak flux depends on the 56Ni mass
            ni56_mass = params[:,6] * params[:,7] * params[:,8] * params[:,9] * params[:,10]
            log_ni56_mass = np.log10(ni56_mass)
            interpolator_max_boundary = self.interpolator_list[12][1](log_ni56_mass)
            interpolator_min_boundary = self.interpolator_list[12][0](log_ni56_mass) 
            params[:,17] = cube[:,17] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## The inner boundary velocity is given by the time, velocity at peak, and velocity gradient for each spectrum
            ## For the first spectrum, the velocity at peak is sampled uniformly, and the velocity gradient is sampled independently
            ## For subsequent spectra, the velocity gradient is sample independently, but the velocity at peak is constrained based on the previous spectrum to ensure that the inner boundary velocity decreases over time
            params[:,18] = cube[:,18] * ( 13000. - 6000. ) + 6000.
            params[:,19] = cube[:,19] * ( 250. - 50. ) + 50.
            phases = (self.observed_spectra_dates - params[:,1][:, None]) / self.redshift_factor

            for i in np.arange(1, len(self.observed_spectra_dates)):
                previous_phase = phases[:,i-1]
                phase = phases[:,i]

                params[:, 19 + i*2] = cube[:, 19 + i*2] * ( 250. - 50. ) + 50.

                max_allowed_velocity_at_peak = params[:,18 + (i-1)*2] - params[:,19 + (i-1)*2] * ( previous_phase - params[:,16] ) + params[:, 19 + i*2] * ( phase - params[:,16] )
                max_allowed_velocity_at_peak[max_allowed_velocity_at_peak > 13000.] = 13000.

                min_allowed_velocity_at_peak = 4000. + params[:, 19 + i*2] * ( phase - params[:,16] )
                min_allowed_velocity_at_peak[min_allowed_velocity_at_peak < 6000.] = 6000.

                params[:,18 + i*2] = cube[:,18 + i*2] * ( max_allowed_velocity_at_peak - min_allowed_velocity_at_peak ) + min_allowed_velocity_at_peak
        elif self.model_type == 'double_detonation':
            ## ***************************
            ## Ejecta structure parameters
            ## ***************************
            ## Shell and core masses are the independent parameters from which many other parameters are derived
            params[:,5] = cube[:,5] * ( 0.10 - 0.03 ) + 0.03

            ## Total ejecta mass is the sum of the shell mass and the core mass
            ## Core mass is uniformly sampled, then the shell mass is added
            params[:,6] = cube[:,6] * ( 1.0 - 0.8 ) + 0.8 + params[:,5]
            log_ejecta_mass = np.log10(params[:,6])

            ## Core fraction depends on shell mass and total ejecta mass
            params[:,7] = (params[:,6] - params[:,5]) / params[:,6]

            ## Core fraction burned depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[3][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[3][0](log_ejecta_mass) 
            params[:,8] = cube[:,8] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ige/burned depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[4][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[4][0](log_ejecta_mass) 
            params[:,9] = cube[:,9] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ni56/ige depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[5][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[5][0](log_ejecta_mass) 
            params[:,10] = cube[:,10] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell fraction burned depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[6][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[6][0](log_ejecta_mass) 
            params[:,11] = cube[:,11] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell fraction ige/burned is uniformly sampled in log space
            params[:,12] = np.power(10, cube[:,12] * ( -0.2 - -2) + -2)

            ## Shell fraction ni56/ige depends on fraction ige/burned
            log_ige_burned = np.log10(params[:,12])
            interpolator_max_boundary = self.interpolator_list[8][1](log_ige_burned)
            interpolator_min_boundary = self.interpolator_list[8][0](log_ige_burned) 
            params[:,13] = np.power(10, cube[:,13] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary)

            ## Shell fraction unburned co depends on fraction burned
            log_burned = np.log10(params[:,11])
            interpolator_max_boundary = self.interpolator_list[9][1](log_burned)
            interpolator_min_boundary = self.interpolator_list[9][0](log_burned)  
            params[:,14] = np.power(10, cube[:,14] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary)

            ## Kinetic energy depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[10][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[10][0](log_ejecta_mass)  
            params[:,15] = cube[:,15] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## ***************************
            ## Simulation input parameters
            ## ***************************
            ## This includes the luminosity and inner boundary velocity for each spectrum
            ## To determine the luminosity, first need to determine a light curve and from that the luminosity at each epoch
            ## The light curve is based on the underlying explosion simulations and scaled to a new rise time and a new peak flux

            ## Rise time depends on the ejecta mass
            interpolator_max_boundary = self.interpolator_list[11][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[11][0](log_ejecta_mass) 
            params[:,16] = cube[:,16] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Peak flux depends on the 56Ni mass
            ni56_mass = params[:,6] * params[:,7] * params[:,8] * params[:,9] * params[:,10]
            log_ni56_mass = np.log10(ni56_mass)
            interpolator_max_boundary = self.interpolator_list[12][1](log_ni56_mass)
            interpolator_min_boundary = self.interpolator_list[12][0](log_ni56_mass) 
            params[:,17] = cube[:,17] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## The inner boundary velocity is given by the time, velocity at peak, and velocity gradient for each spectrum
            ## For the first spectrum, the velocity at peak is sampled uniformly, and the velocity gradient is sampled independently
            ## For subsequent spectra, the velocity gradient is sample independently, but the velocity at peak is constrained based on the previous spectrum to ensure that the inner boundary velocity decreases over time
            params[:,18] = cube[:,18] * ( 13000. - 6000. ) + 6000.
            params[:,19] = cube[:,19] * ( 250. - 50. ) + 50.
            phases = (self.observed_spectra_dates - params[:,1][:, None]) / self.redshift_factor

            for i in np.arange(1, len(self.observed_spectra_dates)):
                previous_phase = phases[:,i-1]
                phase = phases[:,i]

                params[:, 19 + i*2] = cube[:, 19 + i*2] * ( 250. - 50. ) + 50.

                max_allowed_velocity_at_peak = params[:,18 + (i-1)*2] - params[:,19 + (i-1)*2] * ( previous_phase - params[:,16] ) + params[:, 19 + i*2] * ( phase - params[:,16] )
                max_allowed_velocity_at_peak[max_allowed_velocity_at_peak > 13000.] = 13000.

                min_allowed_velocity_at_peak = 4000. + params[:, 19 + i*2] * ( phase - params[:,16] )
                min_allowed_velocity_at_peak[min_allowed_velocity_at_peak < 6000.] = 6000.

                params[:,18 + i*2] = cube[:,18 + i*2] * ( max_allowed_velocity_at_peak - min_allowed_velocity_at_peak ) + min_allowed_velocity_at_peak
        elif self.model_type == 'gravitationally_confined_detonation':
            ## ***************************
            ## Ejecta structure parameters
            ## ***************************
            ## Edefnuc is the independent parameter from which many other parameters are derived
            params[:,5] = cube[:,5] * ( 2.81 - 1.29 ) + 1.29
            log_edefnuc = np.log10(params[:,5])

            ## Total ejecta mass is fixed
            params[:,6] = 1.400486

            ## Core fraction depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[2][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[2][0](log_edefnuc) 
            params[:,7] = cube[:,7] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction burned depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[3][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[3][0](log_edefnuc) 
            params[:,8] = cube[:,8] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ige/burned depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[4][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[4][0](log_edefnuc) 
            params[:,9] = cube[:,9] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ni56/ige depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[5][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[5][0](log_edefnuc) 
            params[:,10] = cube[:,10] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell fraction burned depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[6][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[6][0](log_edefnuc) 
            params[:,11] = cube[:,11] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell fraction ige/burned depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[7][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[7][0](log_edefnuc) 
            params[:,12] = cube[:,12] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Shell fraction ni56/ige depends on fraction ige/burned
            log_ige_burned = np.log10(params[:,11])
            interpolator_max_boundary = self.interpolator_list[8][1](log_ige_burned)
            interpolator_min_boundary = self.interpolator_list[8][0](log_ige_burned)             
            params[:,13] = np.power(10, cube[:,13] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary )

            ## Shell fraction unburned co is fixed to 1
            params[:,14] = 1.

            ## Kinetic energy depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[10][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[10][0](log_edefnuc) 
            params[:,15] = cube[:,15] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## ***************************
            ## Simulation input parameters
            ## ***************************
            ## This includes the luminosity and inner boundary velocity for each spectrum
            ## To determine the luminosity, first need to determine a light curve and from that the luminosity at each epoch
            ## The light curve is based on the underlying explosion simulations and scaled to a new rise time and a new peak flux

            ## Rise time depends on edefnuc
            interpolator_max_boundary = self.interpolator_list[11][1](log_edefnuc)
            interpolator_min_boundary = self.interpolator_list[11][0](log_edefnuc)  
            params[:,16] = cube[:,16] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Peak flux depends on the 56Ni mass
            ni56_mass = params[:,6] * params[:,7] * params[:,8] * params[:,9] * params[:,10]
            log_ni56_mass = np.log10(ni56_mass)
            interpolator_max_boundary = self.interpolator_list[12][1](log_ni56_mass)
            interpolator_min_boundary = self.interpolator_list[12][0](log_ni56_mass) 
            params[:,17] = cube[:,17] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## The inner boundary velocity is given by the time, velocity at peak, and velocity gradient for each spectrum
            ## For the first spectrum, the velocity at peak is sampled uniformly, and the velocity gradient is sampled independently
            ## For subsequent spectra, the velocity gradient is sample independently, but the velocity at peak is constrained based on the previous spectrum to ensure that the inner boundary velocity decreases over time
            params[:,18] = cube[:,18] * ( 13000. - 6000. ) + 6000.
            params[:,19] = cube[:,19] * ( 250. - 50. ) + 50.
            phases = (self.observed_spectra_dates - params[:,1][:, None]) / self.redshift_factor

            for i in np.arange(1, len(self.observed_spectra_dates)):
                previous_phase = phases[:,i-1]
                phase = phases[:,i]

                params[:, 19 + i*2] = cube[:, 19 + i*2] * ( 250. - 50. ) + 50.

                max_allowed_velocity_at_peak = params[:,18 + (i-1)*2] - params[:,19 + (i-1)*2] * ( previous_phase - params[:,16] ) + params[:, 19 + i*2] * ( phase - params[:,16] )
                max_allowed_velocity_at_peak[max_allowed_velocity_at_peak > 13000.] = 13000.

                min_allowed_velocity_at_peak = 4000. + params[:, 19 + i*2] * ( phase - params[:,16] )
                min_allowed_velocity_at_peak[min_allowed_velocity_at_peak < 6000.] = 6000.

                params[:,18 + i*2] = cube[:,18 + i*2] * ( max_allowed_velocity_at_peak - min_allowed_velocity_at_peak ) + min_allowed_velocity_at_peak
        elif self.model_type == 'mergers':
            ## ***************************
            ## Ejecta structure parameters
            ## ***************************
            ## Mass ratio and total ejecta mass are the independent parameters from which many other parameters are derived
            params[:,5] = cube[:,5] * ( 1.0 - 0.8 ) + 0.8

            ## Total ejecta mass is uniformly sampled
            params[:,6] = cube[:,6] * ( 2.00 - 1.67 ) + 1.67
            log_ejecta_mass = np.log10(params[:,6])

            ## Core fraction is fixed to 1. for mergers
            params[:,7] = 1.

            ## Core fraction burned depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[3][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[3][0](log_ejecta_mass) 
            params[:,8] = cube[:,8] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ige/burned depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[4][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[4][0](log_ejecta_mass) 
            params[:,9] = cube[:,9] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Core fraction ni56/ige is uniformly sampled
            params[:,10] = cube[:,10] * ( 0.95 - 0.75 ) + 0.75

            ## Shell parameters are all 0 for mergers
            params[:,11] = 0.
            params[:,12] = 0.
            params[:,13] = 0.
            params[:,14] = 0.

            ## Kinetic energy depends on total ejecta mass
            interpolator_max_boundary = self.interpolator_list[10][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[10][0](log_ejecta_mass) 
            params[:,15] = cube[:,15] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## ***************************
            ## Simulation input parameters
            ## ***************************
            ## This includes the luminosity and inner boundary velocity for each spectrum
            ## To determine the luminosity, first need to determine a light curve and from that the luminosity at each epoch
            ## The light curve is based on the underlying explosion simulations and scaled to a new rise time and a new peak flux

            ## Rise time depends on the ejecta mass
            interpolator_max_boundary = self.interpolator_list[11][1](log_ejecta_mass)
            interpolator_min_boundary = self.interpolator_list[11][0](log_ejecta_mass) 
            params[:,16] = cube[:,16] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## Peak flux depends on the 56Ni mass
            ni56_mass = params[:,6] * params[:,7] * params[:,8] * params[:,9] * params[:,10]
            log_ni56_mass = np.log10(ni56_mass)
            interpolator_max_boundary = self.interpolator_list[12][1](log_ni56_mass)
            interpolator_min_boundary = self.interpolator_list[12][0](log_ni56_mass) 
            params[:,17] = cube[:,17] * ( interpolator_max_boundary - interpolator_min_boundary ) + interpolator_min_boundary

            ## The inner boundary velocity is given by the time, velocity at peak, and velocity gradient for each spectrum
            ## For the first spectrum, the velocity at peak is sampled uniformly, and the velocity gradient is sampled independently
            ## For subsequent spectra, the velocity gradient is sample independently, but the velocity at peak is constrained based on the previous spectrum to ensure that the inner boundary velocity decreases over time
            params[:,18] = cube[:,18] * ( 13000. - 6000. ) + 6000.
            params[:,19] = cube[:,19] * ( 250. - 50. ) + 50.
            phases = (self.observed_spectra_dates - params[:,1][:, None]) / self.redshift_factor

            for i in np.arange(1, len(self.observed_spectra_dates)):
                previous_phase = phases[:,i-1]
                phase = phases[:,i]

                params[:, 19 + i*2] = cube[:, 19 + i*2] * ( 250. - 50. ) + 50.

                max_allowed_velocity_at_peak = params[:,18 + (i-1)*2] - params[:,19 + (i-1)*2] * ( previous_phase - params[:,16] ) + params[:, 19 + i*2] * ( phase - params[:,16] )
                max_allowed_velocity_at_peak[max_allowed_velocity_at_peak > 13000.] = 13000.

                min_allowed_velocity_at_peak = 4000. + params[:, 19 + i*2] * ( phase - params[:,16] )
                min_allowed_velocity_at_peak[min_allowed_velocity_at_peak < 6000.] = 6000.

                params[:,18 + i*2] = cube[:,18 + i*2] * ( max_allowed_velocity_at_peak - min_allowed_velocity_at_peak ) + min_allowed_velocity_at_peak

        return params
        

    def predict_spectra(self, input_parameters):
        """
        Function to generate the neural network predictions for a set of spectra given a set of input parameters.
        This function is used for plotting and so only takes a single set of input parameters and returns the predicted spectra.
        A set of spectra therefore constitutes spectra at different epochs for a single ejecta model.
        This function will take the input parameters, format them, predict the spectra, process the predictions, and return the flux and flux uncertainty for each wavelength bin.
        Spectra are also scaled to some distance and host galaxy extinction is applied.
        """

        ## A number of parameters describing the ejecta structure are constant for all spectra
        common_parameters = np.array([
            input_parameters[5],  # Explosion strength parameter
            input_parameters[6],  # Total ejecta mass
            input_parameters[7],  # Core fraction
            input_parameters[8],  # Core fraction burned
            input_parameters[9],  # Core fraction IGE/burned
            input_parameters[10], # Core fraction Ni56/IGE
            input_parameters[11], # Shell fraction burned
            input_parameters[12], # Shell fraction IGE/burned
            input_parameters[13], # Shell fraction Ni56/IGE
            input_parameters[14], # Shell fraction unburned C/O
            np.power(10, input_parameters[15]) / 1e10,  # KE. There was a bug with the units in the training data when the neural networks were trained, so divide by 1e10 to correct for this
            input_parameters[16],  # Rise time
            np.power(10, input_parameters[17]),  # Peak flux
        ])

        ## Some parameters are specific to each spectrum
        ## These are the time since explosion, velocity at peak, and velocity gradient
        ## ::2 is used to select every second element, corresponding to each new spectrum
        spectrum_specific_parameters = np.column_stack([
            input_parameters[18::2],         # Velocity at peak
            input_parameters[19::2],         # Velocity gradient
            (self.observed_spectra_dates - input_parameters[1]) / self.redshift_factor,  # input_parameters[1] is the explosion date, so this calculates the time since explosion for each spectrum in the rest frame
        ])
    
        ## Stack the common parameters and spectrum-specific parameters together to create the input for the neural network
        all_parameters = np.hstack([
            np.tile(common_parameters, (spectrum_specific_parameters.shape[0], 1)),
            spectrum_specific_parameters
        ])

        ## Log and transform all the parameters for generating the spectra
        all_parameters = np.log10(all_parameters)
        all_parameters[np.isinf(all_parameters)] = -999.
        all_parameters[np.isnan(all_parameters)] = -999.
    
        all_parameters = self.physical_properties_scaler.transform( all_parameters )
        all_parameters = torch.tensor(all_parameters, dtype=torch.float32, device=self.device)

        ## Predict the spectra using the neural network
        ## Use torch.inference_mode() to disable gradient calculations for memory efficiency
        with torch.inference_mode():
            means, stds = self.trained_nn(all_parameters)

        means = means.detach().cpu().numpy().astype(np.float64)
        stds = stds.detach().cpu().numpy().astype(np.float64)

        ## Inverse transform the predicted fluxes and uncertainties
        ## This will put the predictions back into physical flux units
        flux_spectra = np.power(10, self.physical_flux_scaler.inverse_transform(means))
        flux_stds = np.square(flux_spectra) * 5.301898110478399 * np.square(self.physical_flux_scaler.scale_ * stds)
        flux_stds = np.sqrt(flux_stds) * self.uncertainty_scale_factor

        ## Apply observational parameters to the predicted spectrum
        ## Apply distance scaling to the predicted spectra
        ## input_parameters[2] is the distance modulus so convert to cm
        distance = 4. * np.pi * np.square((np.power(10, (input_parameters[2] + 5.) / 5.) * u.pc).to('cm')).value 
        ## Scale the flux spectra and uncertainties to the correct distance
        flux_spectra = flux_spectra / distance
        flux_stds = flux_stds / distance
        ## Apply host galaxy extinction if ebv > 0
        flux_spectra    = apply(fitzpatrick99(self.wavelengths, input_parameters[3] * input_parameters[4], input_parameters[4]), flux_spectra)
        flux_stds       = apply(fitzpatrick99(self.wavelengths, input_parameters[3] * input_parameters[4], input_parameters[4]), flux_stds)

        return flux_spectra, flux_stds


    def predict_spectra_vectorized(self, input_parameters):
        """
        Function to generate the neural network predictions for multiple sets of spectra given multiple sets of input parameters.
        This function is used by UltraNest during fitting and so will take multiple sets of input parameters and return the predicted spectra.
        This function will take the input parameters, format them, predict the spectra, process the predictions, and return the flux and flux uncertainty for each wavelength bin.
        Spectra are also scaled to some distance and host galaxy extinction is applied.
        """    

        ## input_parameters is vectorized, so it contains parameters for N sets of M spectra
        ## Format the input_parameters to be the correct shape for the neural network
        all_parameters = np.zeros((input_parameters.shape[0], len(self.observed_spectra_dates), 16))

        ## A number of parameters describing the ejecta structure are constant for all spectra
        ## These are:
        ## 0, Explosion strength parameter
        ## 1, Total ejecta mass
        ## 2, Core fraction
        ## 3, Core fraction burned
        ## 4, Core fraction IGE/burned
        ## 5, Core fraction Ni56/IGE
        ## 6, Shell fraction burned
        ## 7, Shell fraction IGE/burned
        ## 8, Shell fraction Ni56/IGE
        ## 9, Shell fraction unburned C/O
        ## 10, KE. There was a bug with the units in the training data when the neural networks were trained, so divide by 1e10 to correct for this
        ## 11, Rise time
        ## 12, Peak flux
        all_parameters[:,:,0:13] = input_parameters[:, 5:18][:, np.newaxis, :]
        all_parameters[:, :, 10] = np.power(10, all_parameters[:, :, 10]) / 1e10 # There was a bug with the KE units in the training data when the neural networks were trained, so divide by 1e10 to correct for this.
        all_parameters[:, :, 12] = np.power(10, all_parameters[:, :, 12])

        ## The remaining parameters are specific to each spectrum
        all_parameters[:, :, 13:] = input_parameters[:,18::2][:, :, np.newaxis]  # Velocity at peak
        all_parameters[:, :, 14:] = input_parameters[:,19::2][:, :, np.newaxis]  # Velocity gradient
        all_parameters[:,:,15] = (self.observed_spectra_dates - input_parameters[:,1][:, np.newaxis]) / self.redshift_factor  # input_parameters[1] is the explosion date, so this calculates the time since explosion for each spectrum in the rest frame
        
        ## Reshape the parameters to be the correct shape for the neural network
        all_parameters = all_parameters.reshape((all_parameters.shape[0] * all_parameters.shape[1], all_parameters.shape[-1]))

        ## Log and transform all the parameters for generating the spectra
        all_parameters = np.log10(all_parameters, out=np.full_like(all_parameters, -999.0), where=all_parameters > 0)
        all_parameters = self.physical_properties_scaler.transform( all_parameters )
        all_parameters = torch.tensor(all_parameters, dtype=torch.float32, device=self.device)

        ## Predict the spectra using the neural network
        ## Use torch.inference_mode() to disable gradient calculations for memory efficiency
        with torch.inference_mode():
            means, stds = self.trained_nn(all_parameters)

        means = means.detach().cpu().numpy().astype(np.float64)
        stds = stds.detach().cpu().numpy().astype(np.float64)

        ## Inverse transform the predicted fluxes and uncertainties
        ## This will put the predictions back into physical flux units
        flux_spectra = np.power(10, self.physical_flux_scaler.inverse_transform(means))
        flux_stds = np.square(flux_spectra) * 5.301898110478399 * np.square(self.physical_flux_scaler.scale_ * stds)
        flux_stds = np.sqrt(flux_stds) * self.uncertainty_scale_factor

        flux_spectra = flux_spectra.reshape((input_parameters.shape[0], len(self.observed_spectra_dates), self.n_bins))
        flux_stds = flux_stds.reshape((input_parameters.shape[0], len(self.observed_spectra_dates), self.n_bins))

        ## Apply observational parameters to the predicted spectrum
        ## Apply distance scaling to the predicted spectra
        ## input_parameters[2] is the distance modulus so convert to cm
        distance = 4. * np.pi * np.square((np.power(10, (input_parameters[:,2] + 5.) / 5.) * u.pc).to('cm')).value[:, np.newaxis, np.newaxis]
        ## Scale the flux spectra and uncertainties to the correct distance
        flux_spectra = flux_spectra / distance
        flux_stds = flux_stds / distance

        flux_spectra = np.array([
            apply(fitzpatrick99(self.wavelengths, ebv * rv, rv), spectra) for ebv, rv, spectra in zip(input_parameters[:, 3], input_parameters[:, 4], flux_spectra)
            ])
        flux_stds = np.array([
            apply(fitzpatrick99(self.wavelengths, ebv * rv, rv), stds) for ebv, rv, stds in zip(input_parameters[:, 3], input_parameters[:, 4], flux_stds)
            ])

        return flux_spectra, flux_stds
    

    def log_likelihood(self, input_parameters):
        """
        Function to calculate the log likelihood for multiple sets of input parameters.
        This function is used by UltraNest during fitting and so will take multiple sets of input parameters and return the log likelihood for each set.
        """

        ## The first parameter is log_f, which is the scaling factor representing additional systematic uncertainty
        ## This is a single parameter that applies to all spectra in a set so need to reshape
        log_f = input_parameters[:, 0][:, np.newaxis, np.newaxis]

        ## Generate the predicted spectra for each set of input parameters
        predicted_flux_spectra, predicted_flux_stds = self.predict_spectra_vectorized(input_parameters)

        ## Create an array to hold the log likelihoods for each set of input parameters
        ## By default set to a very large negative number
        log_likelihoods = np.ones( (input_parameters.shape[0]) ) * -1e100

        ## If any flux is outside the range specified below then reject
        ## Have some offset that depends on the flux to avoid numerical issues and plateaus in the likelihood
        mask = (predicted_flux_spectra < -1e41) | (predicted_flux_spectra > 1e41) | (predicted_flux_stds > 1e41)
        points_mask = np.sum(np.sum(mask, axis=-1), axis=-1) > 0
        log_likelihoods[points_mask] = -1e100 * np.abs(np.mean(np.mean(predicted_flux_spectra, axis=1), axis=1))[points_mask]

        ## Get the total uncertainty
        ## Includes the predicted uncertainty, the observational uncertainty, and the additional systematic uncertainty represented by log_f * predicted flux
        sigma2 = np.square(self.observed_flux_errors) + np.square(predicted_flux_stds) + (np.square(predicted_flux_spectra) * np.exp(2 * log_f))

        ## For the accepted points, calculate the likelihood
        ## Include weighting for each wavelength at each epoch
        log_likelihoods[~points_mask] = -0.5 * np.sum(np.sum( ( np.power(self.observed_fluxes - predicted_flux_spectra, 2.0) / sigma2 + np.log(sigma2) + np.log(2.*np.pi) ) * self.observed_flux_weights, axis=1 ), axis=1)[~points_mask]

        return log_likelihoods
