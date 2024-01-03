# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:16:07 2021

@author: cd
"""

import figs_syn as figs
import matplotlib.pyplot as plt

# 1986
fig1986 = figs.fig_manuscript_hcst(
    dt_bgn='1986-02-01 12:00',
    dt_end='1986-02-28 12:00',
    dt_peak='1986-02-18 12:00',
    dt_leads={8: '1986-02-10 12',
              7: '1986-02-11 12',
              6: '1986-02-12 12'},
    ymax_ens=15000,
)
# fig1986.savefig(f'results/figs/SynManuscript_EFO_hcst_1986.png', dpi=600)

# 1995
fig1995 = figs.fig_manuscript_hcst(
    dt_bgn='1994-12-22 12:00',
    dt_end='1995-01-22 12:00',
    dt_peak='1995-01-09 12:00',
    dt_leads={7: '1995-01-02 12:00',
              6: '1995-01-03 12:00',
              5: '1995-01-04 12:00'},
    ymax_ens=15000,
)
# fig1995.savefig(f'results/figs/SynManuscript_EFO_hcst_1995.png', dpi=600)

# 1997
fig1997 = figs.fig_manuscript_hcst(
    dt_bgn='1996-12-10 12:00',
    dt_end='1997-01-10 12:00',
    dt_peak='1997-01-02 12:00',
    dt_leads={9: '1996-12-24 12:00',
              7: '1996-12-26 12:00',
              5: '1996-12-28 12:00'},
    ymax_ens=15000,
)
# fig1997.savefig(f'results/figs/SynManuscript_EFO_hcst_1997.png', dpi=600)

# 1998
fig1998 = figs.fig_manuscript_hcst(
    dt_bgn='1998-02-01 12:00',
    dt_end='1998-02-28 12:00',
    dt_peak='1998-02-20 12:00',
    dt_leads={9: '1998-02-11 12:00',
              6: '1998-02-14 12:00',
              3: '1998-02-17 12:00'},
    ymax_ens=8000,
)
# fig1998.savefig(f'results/figs/SynManuscript_EFO_hcst_1998.png', dpi=600)

# 2006
fig2006 = figs.fig_manuscript_hcst(
    dt_bgn='2005-12-10 12:00',
    dt_end='2006-01-10 12:00',
    dt_peak='2005-12-31 12:00',
    dt_leads={7: '2005-12-24 12:00',
              5: '2005-12-26 12:00',
              3: '2005-12-28 12:00'},
    ymax_ens=15000,
    annotate_ha='right'
)
# fig2006.savefig(f'results/figs/SynManuscript_EFO_hcst_1998.png', dpi=600)
