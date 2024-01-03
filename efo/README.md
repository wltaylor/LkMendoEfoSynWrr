# Ensemble Forecast Operations (EFO)

Private repository for collaboration with the Ensemble Forecast Operations model.

## Environment Setup

To set up the virtual environment for running `EFO` on Comet:

```bash
# Request a compute node
srun --partition=shared --pty -t 08:00:00 --wait=0 --export=ALL /bin/bash

conda create -y -p /cw3e/mead/projects/cwp123/venv_efo

# Save the first 1 MB on Metadata Targets (MDTs) and the rest on Object storage targets (OSTs).
# This is purely for improving performance on Comet.
#
lfs setstripe -E 1M --layout mdt -E -1 /cw3e/mead/projects/cwp123/venv_efo

conda activate /cw3e/mead/projects/cwp123/venv_efo
conda install -y python==3.8

conda install -c conda-forge -y mamba
mamba install -c conda-forge -y \
    scipy h5py pandas numpy matplotlib \
    distributed=2022.10.0 dask=2022.10.0 \
    dask-core=2022.10.0 dask-jobqueue=0.8.1 \
    tqdm=4.62.3 ipywidgets=7.6.5

python -m ipykernel install --user --name venv_efo

# Remove this environment
conda env remove -n venv_efo
jupyter kernelspec uninstall venv_efo
rm -rf /cw3e/mead/projects/cwp123/venv_efo

# Make sure the removal is successful
conda env list
jupyter kernelspec list
```

`EFO` is not currently installed as a third-party module, so you need to include the path to the source code before invoking python.

```bash
export PYTHONPATH=~/github/EFO
```
