## riddler

riddler is a Python code designed to enable automated fitting of type Ia supernovae spectral time series. riddler is comprised of a series of neural networks trained to emulate radiative transfer simulations from [TARDIS](https://github.com/tardis-sn/tardis). Emulated spectra are then fit to observations using nested sampling implemented in [UltraNest](https://johannesbuchner.github.io/UltraNest/readme.html) to estimate the posterior distributions of model parameters and evidences.


## Dependencies

riddler requires a number of dependencies including [SciPy](https://scipy.org/), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [Astropy](https://www.astropy.org/), [UltraNest](https://johannesbuchner.github.io/UltraNest/readme.html), [SpectRes](https://spectres.readthedocs.io/en/latest/), and [extinction](https://extinction.readthedocs.io/en/latest/#). 

The spec file required to make a Conda environment to run riddler on mac OS (ARM) is included within this repository. Note that this also requires [miniforge](https://github.com/conda-forge/miniforge). 


## Usage

Create a folder inside **Inputs** with the name of the supernova to be fit. Spectra are assumed to be .dat files with four columns: wavelength, flux, flux error, and wavelength weight. This folder should also contain a file for the supernova properties, **properties.csv**, including the time of each spectrum, redshift, distance modulus, and extinction (in Av). An example for SN2011fe is included within this repository.

riddler can be run from the command line using
  
  ```python riddler.py SN_NAME MODEL_TYPE RESTART_FLAG NN_INDEX```

where
* ```SN_NAME``` gives the name of the supernova to be fit
* ```MODEL_TYPE``` specifies which type of explosion model will be used during fitting (currently limited to W7 or N100)
* ```RESTART_FLAG``` is an UltraNest option specifying the resume status. More information can be found [here](https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler)
* ```NN_INDEX``` specifies which neural network will be used during the fit. Currently given as an index from 0 - 5

A new folder given by ```SN_NAME``` will be created in the **Outputs** folder and contain the results of the UltraNest run and a quick plot showing the best fitting model spectra compared to the input spectra.
