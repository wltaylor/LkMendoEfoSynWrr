# Lake Mendocino EFO model 
Code repository to support WRR manuscript 'Synthetic forecast ensembles for evaluating Forecast Informed Reservoir Operations '   
Final submission 2 January 2024

## Description
The code below supports simulation from the Ensemble Forecast Operations (EFO) model for various synthetic forecast versions described in the manuscript and the generation of associated figures.
## Getting started
### Dependencies
Releases of this code are stored in the following Zenodo repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10453082.svg)](https://doi.org/10.5281/zenodo.10453082)   
Larger input data files and results are stored in the following Zenodo repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10453068.svg)](https://doi.org/10.5281/zenodo.10453068)   
The data from these files should be downloaded to the 'inp' and 'results' sub-repos respectively 
### Installing
Requires standard Python modules:
* pandas
* numpy
* matplotlib
### Executing program
The workflow below is configured to run from scripts in the root directory. Simulation scripts can be run in any order after downloading required input data from Zenodo. Downloading 'results' folder from Zenodo gives the results of EFO simulations run during the study which can be used to generate the figures.
#### EFO model simulation
##### Repos
- ./efo: Repository for EFO model components
- ./inp: Repository for EFO input files (HEFS or syn-HEFS or syn-GEFS)
- ./results: Repository for EFO outputs; required for plotting routines
##### Scripts
- run_efo_hefs.py: Runs EFO model for the HEFS forecast
- run_efo_syn_hefs.py: Runs EFO model for the syn-HEFS or syn-GEFS forecast samples
- run_efo_syn_hefs_prehcst.py: Runs EFO model using syn-HEFS for the pre-hindcast period
- run_pfo.py: Run EFO model with Perfect Forecast Operations (PFO)

#### Plotting
- manuscript_figs_exc.py: Plotting routines for main manuscript figures
- manuscript_figs_hcst.py: Plotting routines for individual hindcast events with synthetic forecasts
- manuscript_figs_prehcst.py: Plotting routines for individual pre-hindcast events with synthetic forecasts

#### Miscellaneous

#### Contact
Zach Brodeur, zpb4@cornell.edu   
Chris Delaney, chris.delaney@scwa.ca.gov

