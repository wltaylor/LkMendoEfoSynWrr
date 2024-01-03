# -*- coding: utf-8 -*-

#############################################################
# Author: Chris Delaney <chris.delaney@scwa.ca.gov>         #
#                                                           #
#         Sonoma County Water Agency in collaboration with  #
#         Center for Western Weather and Water Extremes     #
#         Scripps Institution of Oceanography               #
#         UC San Diego                                      #
#                                                           #
#         https://github.com/hydrophile                     #
#         https://www.sonomawater.org/                      #
#         https://cw3e.ucsd.edu/                            #
#                                                           #
# Date of Creation: 2020/04/28                              #
#############################################################


import pandas as pd
import numpy as np
from efo.units import UnitsStandard
from efo.qin import Qin, QinLkup, Qloss, QinRouted, QinFcst, QinFcstPerfect
from efo.time import TimeCont, TimeFcst
from efo.junction import Junction, ReservoirJunction, JunctionRegulated
from efo.hypso import Hypso
from efo.evap import EvapLkupTbl
from efo.network import Network, NetworkFcst
from efo.rule import RuleRlsSpecified
from efo.rule_emergency import RuleEmgcCtrlOutlet
from efo.rule_curve import RuleCurve
from efo.reach import ReachConstLag
from efo.rule_compliance import RuleComplianceBase, RuleMaxLkupTblElev, RuleMaxQ, RuleMinQ, RuleMinAnnSched
from efo.rule_downstream import RuleDwnStrmConstLag
from efo.rule_efo import RuleFullEfo
from efo.rule_fcst import RuleFcstReleaseSched
from efo.simulation import Simulation
from russian_river_spec import D1610HydroCondUrr
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)


def get_res_params(dt_bgn, dt_end) -> dict:
    res_params = {
        'inp_q': pd.read_csv('inp/inp_Q_1948-2010.csv',
                             index_col=0, parse_dates=True)[dt_bgn:dt_end],
        'inp_pvp': pd.read_csv('inp/inp_PvpOps_1948-2010.csv',
                               index_col=0, parse_dates=True)[dt_bgn:dt_end],
        'inp_loss': pd.read_csv('inp/inp_UrrLoss.csv'),
        'inp_hypso': pd.read_csv('inp/inp_LkMendo_Hypso.csv'),
        'inp_ctrl_outlet': pd.read_csv('inp/inp_LkMendo_Outlet.csv'),
        'inp_spillway': pd.read_csv('inp/inp_LkMendo_Spillway.csv'),
        'inp_evap': pd.read_csv('inp/inp_LkMendo_EvapRate.csv'),
        'inp_emgc_sch': pd.read_csv('inp/inp_LkMendo_EmgcOps.csv'),
        'inp_rule_curve': pd.read_csv('inp/inp_LkMendo_RuleCurve.csv'),
        'inp_wsc': pd.read_csv('inp/inp_WaterSupplyCond_1948-2010.csv',
                               index_col=0, parse_dates=True)[dt_bgn:dt_end],
        'inp_min_q': pd.read_csv('inp/inp_LkMendo_MinFlowSched.csv'),
        'inp_max_rls_sch': pd.read_csv('inp/inp_LkMendo_MaxRlsSched.csv'),
        'inp_risk_tol': pd.read_csv('inp/inp_RiskTolerance.csv'),
    }
    return res_params


def build_network(name, t_bgn, t_end, t_unit, stor_init, res_params) -> tuple:
    # Create object that keeps track of model time
    t = TimeCont('simTime', t_bgn, t_end, t_unit)
    # Create object that holds model constants
    units = UnitsStandard(t)
    # ---------------LAKE MENDOCINO JUNCTION-----------------
    # Natural inflows
    qin_lm = Qin(name + '_qin_lm', t, res_params['inp_q']['qLkMendocino'].to_numpy())
    # PVP trans-basin diversions
    qin_pvp = Qin(name + '_qin_pvp', t,  res_params['inp_pvp']['qPvpImport'].to_numpy())
    # Define losses upstream and from direct diversions
    loss_month_day_hr =  res_params['inp_loss'][['month', 'day', 'hour']].to_numpy()
    loss_lkup_lm = QinLkup(name + '_loss_lkup_lm', t, loss_month_day_hr,
                           res_params['inp_loss']['lossLkMendocino'].to_numpy(), tbl_type='interp')
    loss_lm = Qloss(name + '_loss_lm', loss_lkup_lm)
    # Lake hypsometry
    hypso_lm = Hypso(
        name + '_hypso_lm',  res_params['inp_hypso']['Elev'].to_numpy(),
        res_params['inp_hypso']['Storage'].to_numpy(),
        res_params['inp_hypso']['Area'].to_numpy())
    # Create juntion object
    res_lm = ReservoirJunction(name + '_res_lm', t, [qin_lm, qin_pvp, loss_lm],
                              stor_init, units, hypso=hypso_lm)
    # Add the controlled outlet rating curve
    res_lm.create_ctrl_outlet(
        name + '_conduit',  res_params['inp_ctrl_outlet']['elev'].to_numpy(),
        res_params['inp_ctrl_outlet']['flow'].to_numpy())
    # Add the uncontrolled outlet rating curve
    res_lm.create_unctrl_outlet(
        name + '_spillway',  res_params['inp_spillway']['elev'].to_numpy(),
        res_params['inp_spillway']['flow'].to_numpy())
    # Set evaporation
    lm_evap = EvapLkupTbl(
        name + '_lm_evap', t, hypso_lm,
        res_params['inp_evap'][['month', 'day', 'hour']].to_numpy(),
        res_params['inp_evap']['evapRate'].to_numpy(), time_unit='M')
    res_lm.set_evap(lm_evap)
    # Create and set emergency release rule
    rule_lm_emgc = RuleEmgcCtrlOutlet(
        name + '_rule_lm_emgc', t, res_lm, units,
        elev=res_params['inp_emgc_sch']['elev'].to_numpy(),
        rls_emgc=res_params['inp_emgc_sch']['flow'].to_numpy())
    # Create arrays for min flows and buffers
    mon_day_hr_min_q = res_params['inp_min_q'][['month', 'day', 'hour']].to_numpy()
    min_q_sch = res_params['inp_min_q'][
        ['minQ_normal', 'minQ_DS1', 'minQ_DS2', 'minQ_dry', 'minQ_critical']].to_numpy()
    buffer_urr = res_params['inp_min_q'][
        ['buffer_normal', 'buffer_DS1', 'buffer_DS2', 'buffer_dry', 'buffer_critical']].to_numpy()
    buffer_forks = 8
    hydro_cond_urr = D1610HydroCondUrr(
        name + '_hydro_cond_urr', t, res_lm,
        res_params['inp_wsc']['waterSupplyCond'].to_numpy(),
        res_params['inp_pvp']['storLkPillsbury'].to_numpy())
    # ---------------------FORKS JUNCTION----------------------------
    # West Fork flow
    qin_wf = Qin(name + '_qin_wf', t, res_params['inp_q']['qWestFork'].to_numpy())
    # Define reach from Lake Mendocino to Forks
    reach_lm2forks = ReachConstLag('_reach_lm2forks', t, res_lm, 0)
    rls_from_lm = QinRouted(name + '_rls_from_lm', t, reach_lm2forks)
    # Create Forks min flow rule
    rule_min_q_forks = RuleMinAnnSched(
        name + '_rule_min_q_forks', t, mon_day_hr_min_q, min_q_sch + buffer_forks,
        time_unit='D', hydro_cond=hydro_cond_urr)
    jnc_forks = JunctionRegulated(
        name + '_jnc_forks', t, [qin_wf, rls_from_lm], rule_min_q_forks)
    # Upper Russian Min Flow Rule
    rule_min_q_urr = RuleMinAnnSched(
        name + '_rule_min_q_urr', t, mon_day_hr_min_q, min_q_sch + buffer_urr,
        time_unit='D', hydro_cond=hydro_cond_urr)
    # -------------------HOPLAND JUNCTION-------------------------
    # Hopland local flow
    qin_local_hop = Qin(name + '_qin_local_hop', t, res_params['inp_q']['qHopland'].to_numpy())
    # Define Hopland reach losses
    loss_lkup_hop = QinLkup(
        name + '_loss_lkup_hop', t, loss_month_day_hr,
        res_params['inp_loss']['lossHopland'].to_numpy(), tbl_type='interp')
    loss_hop = Qloss(name + '_loss_hop', loss_lkup_hop)
    # Define reach from Forks to Hopland
    reach_forks2hop = ReachConstLag(name + '_reach_forks2hop', t, jnc_forks, 0)
    q_routed_from_forks = QinRouted(name + '_q_routed_from_forks', t, reach_forks2hop)
    # Max 8K cfs rule
    rule_hop_max = RuleMaxQ(name + '_rule_hop_max', t, 8000, pct_buffer=0.2)
    # Create Hopland junction
    jnc_hop = JunctionRegulated(
        name + '_jnc_hop', t,
        [qin_local_hop, q_routed_from_forks, loss_hop], rule_min_q_urr, rule_hop_max)
    # ------------------CLOVERDALE JUNCTION---------------------
    # Cloverdale local flow
    qin_local_clov = Qin(
        name + '_qin_local_clov', t, res_params['inp_q']['qCloverdale'].to_numpy())
    # Define Cloverdale reach losses
    loss_lkup_clov = QinLkup(
        name + '_loss_lkup_clov', t, loss_month_day_hr,
        res_params['inp_loss']['lossCloverdale'].to_numpy(), tbl_type='interp')
    loss_clov = Qloss(name + '_loss_clov', loss_lkup_clov)
    # Define reach from Hopland to Cloverdale
    reach_hop2clov = ReachConstLag(name + '_reach_hop2clov', t, jnc_hop, 1)
    q_routed_from_hop = QinRouted(name + '_q_routed_from_hop', t, reach_hop2clov)
    # Create Cloverdale junction
    jnc_clov = JunctionRegulated(name + '_jnc_clov', t, [qin_local_clov, q_routed_from_hop, loss_clov], rule_min_q_urr)
    # ------------------HEALDSBURG JUNCTION---------------------
    # Healdsburg local flow
    qin_local_hlds = Qin(
        name + '_qin_local_hlds', t, res_params['inp_q']['qHealdsburg'].to_numpy())
    # Define Healdsburg reach losses
    loss_lkup_hlds = QinLkup(
        name + '_loss_lkup_hlds', t, loss_month_day_hr,
        res_params['inp_loss']['lossHealdsburg'].to_numpy(), tbl_type='interp')
    loss_hlds = Qloss(name + '_loss_hlds', loss_lkup_hlds)
    # Define reach from Cloverdale to Healdsburg
    reach_clov2hlds = ReachConstLag(name + '_reach_clov2hlds', t, jnc_clov, 2)
    q_routed_from_clov = QinRouted(name + '_q_routed_from_clov', t, reach_clov2hlds)
    # Create Healdsburg junction
    jnc_hlds = JunctionRegulated(
        name + '_jnc_hlds', t, [qin_local_hlds, q_routed_from_clov, loss_hlds], rule_min_q_urr)
    # -------------DOWNSTREAM MIN FLOW RULES--------------------
    # Create downstream min flow rule
    rule_ds_min_q = RuleDwnStrmConstLag(name + '_rule_ds_min_q', t,
                                     [res_lm, jnc_forks, jnc_hop, jnc_clov, jnc_hlds],
                                     rule_type=RuleComplianceBase.MIN)
    # Create max flow schedule rule
    # Create min release rule
    rule_min_rls = RuleMinQ(name + '_rule_min_rls', t, 25)
    rule_list = [rule_ds_min_q, rule_min_rls, rule_lm_emgc]
    # Add rule stack to res_lm
    # -------------------NETWORK----------------------------
    network = Network(
        name + '_network' + name, t, [res_lm, jnc_forks, jnc_hop, jnc_clov, jnc_hlds])
    return network, units, rule_list


def build_scenario_baseline(*, name, t_bgn, t_end, t_unit, stor_init, res_params) -> Simulation:
    network, units, rule_list = build_network(
        name, t_bgn, t_end, t_unit, stor_init, res_params)
    jnc_lm = network.junctions[0]
    jnc_forks = network.junctions[1]
    jnc_hop = network.junctions[2]
    rule_curve = RuleCurve(
        name + '_LmRuleGuideCurve', network.T, jnc_lm, units,
        res_params['inp_rule_curve'][['month', 'day', 'hour']].to_numpy(),
        res_params['inp_rule_curve']['guide_curve'].to_numpy(), is_elev=False)
    # Create max flow schedule rule
    rule_max_rls_sch = RuleMaxLkupTblElev(
        name + '_LmMaxRlsSch', network.t, jnc_lm,
        res_params['inp_max_rls_sch']['elev'].to_numpy(),
        res_params['inp_max_rls_sch']['maxRls'].to_numpy())
    # Create Downstream Hopland 8K rule
    ruleDsMaxHopQ = RuleDwnStrmConstLag(
        name + '_ruleDsMaxHopQ', network.t, [jnc_lm, jnc_forks, jnc_hop],
        rule_type=RuleComplianceBase.MAX)
    rule_list = [rule_curve, rule_max_rls_sch, ruleDsMaxHopQ] + rule_list
    jnc_lm.set_rule_stack(rule_list)
    simulation = Simulation(name, network)
    return simulation


def build_efo_rule(
        name, network, units, stor_init, res_params,
        hcst_lm, hcst_wf, hcst_hop, fcst_horiz) -> RuleFullEfo:
    if isinstance(res_params['inp_risk_tol'], pd.DataFrame):
        inp_risk_tol = res_params['inp_risk_tol'].to_numpy()
    # Check if EFO or PFO simulation
    n_mbrs = 1 if hcst_lm.ndim == 1 else hcst_lm.shape[2]
    res_lm = network.junctions[0]
    qin_pvp = res_lm.qin[1]
    loss_lm = res_lm.qin[2]
    hypso_lm = res_lm.hypso
    jnc_forks = network.junctions[1]
    jnc_hop = network.junctions[2]
    # Create forecast time object
    t_fcst = TimeFcst(name + '_Tfcst', network.t, fcst_horiz=fcst_horiz)
    # Define Hopland reach losses
    loss_lkup_hop_fcst = QinLkup(
        name + '_lossLkupHopFcst', t_fcst,
        res_params['inp_loss'][['month', 'day', 'hour']].to_numpy(),
        res_params['inp_loss']['lossHopland'].to_numpy(), tbl_type='interp')
    loss_hop_fcst = Qloss(name + '_lossHop', loss_lkup_hop_fcst)
    # Max 8K cfs rule
    pct_buffer = 0.2 if n_mbrs > 1 else 0.
    rule_hop_max_fcst = RuleMaxQ(
        name + '_ruleHopMaxFcst', t_fcst, 8000, pct_buffer=pct_buffer)
    qin_fcst_lm = []
    qin_fcst_wf = []
    qin_fcst_hop = []
    efo_res_lm = []
    efo_jnc_forks = []
    efo_jnc_hop = []
    efo_rls_routed_from_lm = []
    efo_q_routed_from_forks = []
    efo_reach_lm2forks = []
    efo_reach_forks2hop = []
    rule_lm_rls_spec = []
    rule_max_rls_sch_efo = []
    rule_ds_max_hop_q_efo = []
    efo_sched = []
    net_efo = []
    for i in range(0, n_mbrs):
        if n_mbrs > 1:
            cur_qin_fcst_lm = QinFcst(name + '_qInFcstLm_' + str(i), t_fcst, hcst_lm[:, :, i])
            cur_qin_fcst_wf = QinFcst(name + '_qInFcstWf_' + str(i), t_fcst, hcst_wf[:, :, i])
            cur_qin_fcst_hop = QinFcst(name + '_qInFcstHop_' + str(i), t_fcst, hcst_hop[:, :, i])
        else:
            cur_qin_fcst_lm = QinFcstPerfect(name + '_qInFcstLm_Pcst', t_fcst, hcst_lm)
            cur_qin_fcst_wf = QinFcstPerfect(name + '_qInFcstWf_Pcst', t_fcst, hcst_wf)
            cur_qin_fcst_hop = QinFcstPerfect(name + '_qInFcstHop_Pcst', t_fcst, hcst_hop)
        qin_fcst_lm.append(cur_qin_fcst_lm)
        qin_fcst_wf.append(cur_qin_fcst_wf)
        qin_fcst_hop.append(cur_qin_fcst_hop)
        # Create efo reservoir
        rule_lm_rls_spec.append(RuleRlsSpecified(name + '_ruleLmRlsSpec_' + str(i), t_fcst))
        efo_res_lm.append(ReservoirJunction(
            name + '_jncEfoLm_' + str(i), t_fcst, [qin_fcst_lm[i], qin_pvp, loss_lm],
            stor_init, units, hypso=hypso_lm, rules=[rule_lm_rls_spec[i]]))
        efo_res_lm[i].create_ctrl_outlet(
            name + '_ConduitFcst' + str(i),
            res_params['inp_ctrl_outlet']['elev'].to_numpy(),
            res_params['inp_ctrl_outlet']['flow'].to_numpy())
        rule_max_rls_sch_efo.append(RuleMaxLkupTblElev(
            name + '_ruleMaxRlsSch', t_fcst, efo_res_lm[i],
            res_params['inp_max_rls_sch']['elev'].to_numpy(),
            res_params['inp_max_rls_sch']['maxRls'].to_numpy()))
        # FORKS
        # Define reach from Lake Mendocino to Forks
        efo_reach_lm2forks.append(ReachConstLag(
            name + '_reachFcstLm2Forks_' + str(i), t_fcst, efo_res_lm[i], 0))
        efo_rls_routed_from_lm.append(QinRouted(
            name + '_rlsRoutedFromLmFcst_' + str(i), t_fcst, efo_reach_lm2forks[i]))
        efo_jnc_forks.append(Junction(
            name + '_jncEfoForks_' + str(i), t_fcst,
            [qin_fcst_wf[i], efo_rls_routed_from_lm[i]]))
        # HOPLAND
        # Define reach from Forks to Hopland
        efo_reach_forks2hop.append(ReachConstLag(
            name + '_reachFcstForks2Hop_' + str(i), t_fcst, efo_jnc_forks[i], 0))
        efo_q_routed_from_forks.append(QinRouted(
            name + '_qRoutedFromForksFcst_' + str(i), t_fcst, efo_reach_forks2hop[i]))
        # Create Hopland junction
        efo_jnc_hop.append(JunctionRegulated(
            name + '_jncEfoHop_' + str(i), t_fcst,
            [qin_fcst_hop[i], efo_q_routed_from_forks[i], loss_hop_fcst],
            None, rule_hop_max_fcst))
        # Create a list of networks for the EFO rule
        net_efo.append(NetworkFcst(
            name + '_netEfo_' + str(i), t_fcst,
            [res_lm, jnc_forks, jnc_hop],
            [efo_res_lm[i], efo_jnc_forks[i], efo_jnc_hop[i]]))
        # Add the downstream flow rule
        rule_ds_max_hop_q_efo.append(RuleDwnStrmConstLag(
            name + '_ruleDsMaxHopQEfo_' + str(i), t_fcst,
            [efo_res_lm[i], efo_jnc_forks[i], efo_jnc_hop[i]],
            rule_type=RuleComplianceBase.MAX))
        efo_res_lm[i].append_rule([rule_ds_max_hop_q_efo[i], rule_max_rls_sch_efo[i]])
        efo_sched.append(
            RuleFcstReleaseSched(
                f'efoSched_{i}', t_fcst, efo_res_lm[i], net_efo[i], units,
                stor_max=116500., rule_release_specified=rule_lm_rls_spec[i],
                max_itr=100, conv_crit=1.))
    rule_efo = RuleFullEfo(
        name='ruleEfo',
        time_fcst=t_fcst,
        efo_resvr=efo_res_lm,
        stor_thr=111000.,
        risk_tol=inp_risk_tol,
        units=units,
        rule_release_specified=rule_lm_rls_spec,
        efo_release_scheduler=efo_sched,
        operating_junction=res_lm)
    return rule_efo


def build_scenario_efo(*, name, t_bgn, t_end, t_unit, stor_init, res_params,
                       hcst_lm, hcst_wf, hcst_hop, fcst_horiz) -> Simulation:
    network, units, rule_list = build_network(
        name, t_bgn, t_end, t_unit, stor_init, res_params)
    # Build EFO Rule
    rule_efo = build_efo_rule(
        name, network, units, stor_init, res_params,
        hcst_lm, hcst_wf, hcst_hop, fcst_horiz)
    jnc_lm = network.junctions[0]
    rule_list.insert(0, rule_efo)
    jnc_lm.set_rule_stack(rule_list)
    simulation = Simulation(name, network)
    return simulation


def create_results_df(*, simulation, name) -> pd.DataFrame:
    # Create scenario index
    idx_name = np.full(simulation.network.t.datetimes.size, name)
    # Create hypsometry object to convert storage to elevation
    hypso = simulation.network.junctions[0].hypso
    # Create results dictionary
    results_dict = {
        'stor_mendo': simulation.network.junctions[0].stor,
        'elev_mendo': hypso.stor2elev(simulation.network.junctions[0].stor),
        'rls_ctrl': simulation.network.junctions[0].rls_ctrl,
        'rls_tot': simulation.network.junctions[0].qout,
        'rls_max': simulation.network.junctions[0].rule_stack.rls_max,
        'spill': simulation.network.junctions[0].rls_unctrl,
        'q_hop': simulation.network.junctions[2].qout,
        'q_clov': simulation.network.junctions[3].qout,
        'q_hlds': simulation.network.junctions[4].qout,
    }
    # Create a multi-index
    df_indices = [idx_name, simulation.network.t.datetimes]
    df_multi_idx = pd.MultiIndex.from_arrays(df_indices, names=('name_scenario', 'date_time'))
    # Create results dataframe
    results_df = pd.DataFrame(data=results_dict, index=df_multi_idx)
    return results_df

