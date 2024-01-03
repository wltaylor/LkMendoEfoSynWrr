# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:16:07 2021

@author: cd
"""

import pandas as pd
import numpy as np
import figs_syn as figs
import matplotlib.pyplot as plt
from datetime import datetime as dt
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

# 1955
fig1955 = figs.fig_manuscript_prehcst(
    dt_bgn='1955-12-11',
    dt_end='1956-01-05',
    dt_peak='1955-12-22 12:00',
    dt_leads={7: '1955-12-15 12',
              5: '1955-12-17 12',
              3: '1955-12-19 12'},
    ymax_ens=1000. * 20 / 1.983471)
# fig1955.savefig(f'results/figs/SynManuscript_EFO_prehcst_1955.png', dpi=600)

# 1964
fig1964 = figs.fig_manuscript_prehcst(
    dt_bgn='1964-12-08',
    dt_end='1965-01-10',
    dt_peak='1964-12-23 12:00',
    dt_leads={7: '1964-12-16 12',
            5: '1964-12-18 12',
            3: '1964-12-20 12'},
    ymax_ens=1000. * 121 / 1.983471)
# fig1964.savefig(f'results/figs/SynManuscript_EFO_prehcst_1964.png', dpi=600)

# 1970
fig1970 = figs.fig_manuscript_prehcst(
    dt_bgn='1970-01-09',
    dt_end='1970-02-04',
    dt_peak='1970-01-24 12:00',
    dt_leads={7: '1970-01-17 12',
            5: '1970-01-19 12',
            3: '1970-01-21 12'},
    ymax_ens=1000. * 81 / 1.983471)
# fig1970.savefig(f'results/figs/SynManuscript_EFO_prehcst_1964.png', dpi=600)

