## riddler

riddler is a Python code designed to enable automated fitting of type Ia supernovae spectral time series. riddler is comprised of a series of neural networks trained to emulate radiative transfer simulations from [TARDIS](https://github.com/tardis-sn/tardis). Emulated spectra are then fit to observations using nested sampling implemented in [UltraNest](https://johannesbuchner.github.io/UltraNest/readme.html) to estimate the posterior distributions of model parameters and evidences.



## Dependencies

riddler requires a number of dependencies including:
* [Astropy](https://www.astropy.org/)
* [extinction](https://extinction.readthedocs.io/en/latest/#) 
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/install/)
* [pandas](https://pandas.pydata.org/getting_started.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [SciPy](https://scipy.org/)
* [SpectRes](https://spectres.readthedocs.io/en/latest/) 
* [UltraNest](https://johannesbuchner.github.io/UltraNest/readme.html) 



## Usage

Create a folder inside **Inputs/Objects** with the name of the supernova to be fit, ```SN_NAME```. Inside this folder, place the spectra to be fit as **.dat** files. Each spectrum file should contain four columns: wavelength, flux, flux error, and the weighting for that wavelength. Fitting with riddler requires that spectra have been flux calibrated, but they do not need to be corrected for redshift or reddening.

Inside the ```SN_NAME``` folder, there should also be two files. **SN_NAME_times.csv** gives the dates at which the spectra to be fit were observed (JD or MJD). **SN_NAME_properties.csv** contains a number of properties of the supernova that are used by UltraNest during fitting. This includes:
* Redshift
* Distance modulus
* Distance modulus uncertainty
* Milky Way E(B-V)
* Milky Way R_V
* Host galaxy E(B-V)
* Host galaxy E(B-V) uncertainty
* Host galaxy R_V
* Host galaxy R_V uncertainty
* Minimum explosion date to fit
* Maximum explosion date to fit
For those parameters with uncertainties UltraNest will sample new values during fitting from normal distributions given the value and uncertainty input by the user. If no uncertainty is provided, UltraNest fixes this parameter to the value given. The host galaxy extinction can also be drawn from an exponential distribution by setting E(B-V)_host = E(B-V)_host_err = -999. The default Rv_host is 3.1, which is used if no valid value is given. The redshift, Milky Way E(B-V), and Milky Way R_V are all fixed based on the values provided. The explosion date is uniformly sampled between the minimum and maximum dates provided.

An example input folder for SN2011fe is included within this repository.


riddler can be run from the command line using
  
  ```python riddler.py SN_NAME MODEL_TYPE RESTART_FLAG```

where
* ```SN_NAME``` gives the name of the supernova to be fit and the folder name where data is stored.
* ```MODEL_TYPE``` specifies which type of explosion model will be used during fitting. Valid options are DEF, DDT, DOD, GCD, and VM. For further details see [Magee (2026a)]().
* ```RESTART_FLAG``` is an UltraNest option specifying the resume status. More information can be found [here](https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler). If ```false```, a new run is started from scratch. If ```true```, the current run will continue from the last run in the outputs folder.


A new folder given by ```SN_NAME``` will be created in the **Outputs** folder and contain the results of the UltraNest run. Subsequent runs can also be found in this folder. A number of outputs are created for each run, details of which can be found in the UltraNest [documentation](https://johannesbuchner.github.io/UltraNest/performance.html#output-files). In addition, the **plots** folder contains a quick plot showing the best fitting model spectra compared against the input spectra. Some of the parameters fit by riddler are degenerate, therefore the posteriors found by UltraNest are processed and output into a new file in the **chains** folder. This file contains, for example, the 56Ni mass of the models, phases of the spectra, etc. and should be used for analysis. For further details see [Magee (2026a)]().



## Acknowledgement

If you make use of riddler, please cite [Magee et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.3042M/abstract) and [Magee (2026a)]() for the initial and updated releases of the riddler code. Please also cite [TARDIS](https://github.com/tardis-sn/tardis) and [UltraNest](https://johannesbuchner.github.io/UltraNest/readme.html) for the training data generation and nested sampling, respectively. 

