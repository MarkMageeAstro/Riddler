import os
import glob

import numpy as np
import pandas as pd

from extinction import ccm89, fitzpatrick99, remove

import spectres



class ObservedData:
    """
    Class to handle reading and storing observed data that will be fit.
    """
    def __init__(self, root_dir, sn_name, wavelengths):
        """
        Initializes observed data class.

        Parameters
        ----------
        root_dir : str
            The root directory where the data and outputs are stored.
        sn_name : str
            The name of the supernova being fit.
        wavelengths : array-like : float
            The wavelength grid of the neural network.
        """

        self.root_dir = root_dir
        self.sn_name = sn_name
        self.wavelengths = wavelengths

        self.read_observed_properties()
        self.read_observed_times()

        self.read_observed_data()

        return None

    def read_observed_properties(self):
        """
        Function to read in the observed properties for the supernova being fit.
        The observed properties are stored in a .csv file and contain things like the redshift, explosion time range, etc.
        """

        ## Read in the observed properties from a .csv file
        properties_path = self.root_dir + "Inputs/Objects/" + self.sn_name + "/" + self.sn_name + "_properties.csv"
        self.observed_properties = pd.read_csv(properties_path)
        self.observed_properties = self.observed_properties.iloc[0]

        return None

    def read_observed_times(self):
        """
        Function to read in the observed times for the supernova being fit.
        The observed times are stored in a .csv file and contain the dates of the observed spectra.
        """

        ## Read in the observed times from a .csv file
        times_path = self.root_dir + "Inputs/Objects/" + self.sn_name + "/" + self.sn_name + "_times.csv"
        df = pd.read_csv(times_path)
        self.observed_spectra_dates = df['Time'].values

        return None

    def read_observed_data(self):
        """
        Function to read in the observed data for the supernova being fit.
        """

        ## Get a list of the observed spectra
        data_list = sorted(glob.glob(self.root_dir + "Inputs/Objects/" + self.sn_name + "/*.dat"))

        ## Create empty lists to store the fluxes, flux errors, and flux weights for each spectrum
        flux_list = []
        flux_error_list = []
        flux_weight_list = []

        ## Go through the list of spectra
        ## Put each one into rest frame and correct for MW extinction.
        ## Bin to the correct wavelength grid
        i = 0
        for spectrum in data_list:
            print("Reading in spectrum " + str(i+1) + " of " + str(len(data_list)) + ": " + spectrum, flush=True)

            spectrum_df = pd.read_csv(spectrum)

            ## Put into restframe
            spectrum_df['rest_wavelength'] = spectrum_df['wavelength'] / (1 + self.observed_properties['Redshift'])
            print("Wavelength range: " + "{:.2f}".format(spectrum_df['rest_wavelength'].min()) + " - " + "{:.2f}".format(spectrum_df['rest_wavelength'].max()), flush=True)

            ## If no flux error is given then automatically set to 2%
            if 'flux_err' not in spectrum_df.keys():
                print("No flux error column found in spectrum %d. Setting flux errors to 2%% of the flux values.", i+1, flush=True)
                spectrum_df['flux_err'] = 0.02 * spectrum_df['flux']

            ## If any NaNs are present in the flux, flux error, or weight columns, raise an error
            if np.any(np.isnan(spectrum_df['flux'])) | np.any(np.isnan(spectrum_df['flux_err'])) | np.any(np.isnan(spectrum_df['weight'])):
                raise Exception("Error: Spectrum file contains NaNs")

            ## Correct for MW extinction
            ## Extinction expects Rv, not EBV, so convert
            spectrum_df['flux']     = remove(fitzpatrick99(spectrum_df['wavelength'].values, self.observed_properties['E(B-V)_MW'] * self.observed_properties['Rv_MW'], self.observed_properties['Rv_MW']), spectrum_df['flux'].values)
            spectrum_df['flux_err'] = remove(fitzpatrick99(spectrum_df['wavelength'].values, self.observed_properties['E(B-V)_MW'] * self.observed_properties['Rv_MW'], self.observed_properties['Rv_MW']), spectrum_df['flux_err'].values)

            ## Bin to the correct wavelength grid
            binned_flux = spectres.spectres(self.wavelengths, spectrum_df['rest_wavelength'].values, spectrum_df['flux'].values, fill = 0)
            binned_flux_err = spectres.spectres(self.wavelengths, spectrum_df['rest_wavelength'].values, spectrum_df['flux_err'].values, fill = 0)
            binned_weights = spectres.spectres(self.wavelengths, spectrum_df['rest_wavelength'].values, spectrum_df['weight'].values, fill = 0)

            ## If the spectrum does not cover the full wavelength grid, set weights of outside values to 0
            mask = (self.wavelengths < spectrum_df['rest_wavelength'].values[0]) | (self.wavelengths > spectrum_df['rest_wavelength'].values[-1])
            binned_weights[mask] = 0.0

            ## Append the binned flux, flux error, and weights to the lists
            flux_list.append(binned_flux)
            flux_error_list.append(binned_flux_err)
            flux_weight_list.append(binned_weights)

            i += 1
        
        print("Finished reading in " + str(i) + " spectra.", flush=True)

        ## Convert the lists to arrays and store as attributes
        self.observed_fluxes = np.array(flux_list)
        self.observed_flux_errors = np.array(flux_error_list)
        self.observed_flux_weights = np.array(flux_weight_list)

        return None
