# Lake Mendocino EFO model 
Code repository to support WRR manuscript 'Synthetic forecast ensembles for evaluating Forecast Informed Reservoir Operations '   
Final submission 2 January 2024

## Description
The code below supports simulation from the Ensemble Forecast Operations (EFO) model for various synthetic forecast versions described in the manuscript and the generation of associated figures.
## Getting started
### Dependencies
Releases of this code are stored in the following Zenodo repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7702354.svg)](https://doi.org/10.5281/zenodo.7702354)   
Larger input data files and results are stored in the following Zenodo repository:   
The data from these files should be downloaded to the 'inp' and 'results' sub-repos respectively 
### Installing
Requires standard Python modules:
* pandas
* numpy
* matplotlib
### Executing program
The workflow below is configured to run from scripts in the root directory. Simulation scripts can be run in any order after downloading required input data from Zenodo. Downloading 'results' folder from Zenodo gives the results of EFO simulations run during the study which can be used to generate the figures.
#### EFO model simulation

1) src/data_process.R: Processes raw hydrologic and state variable data for HYMOD 'process' model from .txt files in data repository; outputs to 'data' repo 
2) src/fit_model_hymod.R: Fits hybrid SWM model to calibration-validation subsets of training data for HYMOD; outputs to 'fit' repo 
3) src/fit_model_sacsma.R: Fits static SWM model to calibration-validation subsets of training data for SAC-SMA; outputs to 'fit' repo 

#### Plotting
- ms: R Script for plotting primary manuscript figures arranged by figure number
- si: R scripts for plotting supporting information figures arranged by figure number
- figs: Output repository for generated figures

#### Miscellaneous

#### Contact
Zach Brodeur, zpb4@cornell.edu

