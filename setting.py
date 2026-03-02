import numpy as np
from scipy.spatial import distance

# -----------------------------
# Selected wells (seed2026)
# -----------------------------
SEED = 2026
SELECTED_WELLS_CSV = r"data/Stanford_VI/selected_wells_20_seed2026.csv"

TCN1D_train_p = {
    'batch_size': 4,
    'no_wells': 20,
    'unsupervised_seismic': -1,
    'lambda_ai': 5.0,
    'lambda_facies': 0.2,
    'lambda_recon': 1.0,
    # Step-2: two-stage loss scheduling + amplitude anchor
    'use_two_stage_loss': True,
    'stageA_epochs': 200,
    'stageB_ramp_epochs': 200,
    'stageA_lambda_ai_mult': 1.8,
    'stageA_lambda_facies_mult': 0.30,
    'stageA_lambda_recon_mult': 0.30,
    'lambda_amp_anchor': 0.05,
    'facies_detach_y': False,
    'SGS_data_n': 1000,
    'epochs': 1000,
    'grad_clip': 1.0,
    'input_shape': (1, 200),
    'lr': 0.0001,
    # DataLoader
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True,
    # Weak-supervision scheduling
    'ws_every': 2,
    'ws_max_batches': 150,
    'ws_max_batches_stageA': 50,
    'seed': SEED,
    'selected_wells_csv': SELECTED_WELLS_CSV,

    # Facies-adaptive anisotropic conditioning
    'use_aniso_conditioning': True,
    'aniso_train_protocol': 'closed_loop',  # closed_loop | oracle
    'aniso_closed_loop_alpha_prior': 0.0,
    'aniso_closed_loop_conf_thresh': 0.60,
    'agpe_neutral_p': 0.5,
    'channel_id': 2,
    'aniso_steps_R': 25,
    'aniso_eta': 0.6,
    'aniso_gamma': 8.0,
    'aniso_tau': 0.6,
    'aniso_kappa': 4.0,
    'aniso_sigma_st': 1.2,
    'aniso_backend': 'grid',                # grid | graph_lattice | skeleton_graph
    # default False for parity: graph_lattice ~= grid
    'aniso_use_tensor_strength': False,
    'aniso_tensor_strength_power': 1.0,
    # skeleton graph controls
    'agpe_skel_p_thresh': 0.55,
    'agpe_skel_min_nodes': 30,
    'agpe_skel_snap_radius': 5,
    'agpe_long_edges': True,
    'agpe_long_max_step': 6,
    'agpe_long_step': 2,
    'agpe_long_cos_thresh': 0.70,
    'agpe_long_weight': 0.35,
    'agpe_edge_tau_p': 0.25,
    'agpe_lift_sigma': 2.2,
    'agpe_well_seed_mode': 'depth_gate',  # hard | depth_gate
    'agpe_well_seed_power': 1.0,
    'agpe_well_seed_min': 0.02,
    'agpe_well_seed_use_conf': True,
    'agpe_well_soft_alpha': 0.20,
    'agpe_cache_graph': True,
    'agpe_refine_graph': True,
    'agpe_rebuild_every': 50,
    'agpe_topo_change_pch_l1': 0.05,

    # optional soft impedance prior
    'use_soft_prior': False,
    'aniso_steps_prior': 35,
    'lambda_prior': 0.20,

    # Iterative R coupling
    'iterative_R': True,
    'R_update_every': 50,
    'R_update_bs': 16,
    'R_ema_beta': 0.85,
    'alpha_prior_start': 1.0,
    'alpha_prior_end': 0.3,
    'alpha_prior_decay_epochs': 800,
    'conf_thresh': 0.75,
    'lambda_phys_damp': 0.8,
    'save_R_every': 50,

    # model config
    'model_name': 'VishalNet',
    'Forward_model': 'cov_para',
    'Facies_model': 'Facies',
    'run_id_suffix': '',

    # data config
    'data_flag': 'Stanford_VI',
    'get_F': 0,
    'F': 'WE_PreSDM',
}

TCN1D_test_p = {
    'no_wells': 20,
    'seed': SEED,
    'selected_wells_csv': SELECTED_WELLS_CSV,
    'data_flag': 'Stanford_VI',
    # Keep as legacy default; test script will resolve from run_id + run_id_suffix first.
    'model_name': 'VishalNet_cov_para_Facies_s_uns',
    'run_id': 'VishalNet_cov_para_Facies',
    'run_id_suffix': '',
    'use_aniso_conditioning': True,
    'aniso_test_protocol': 'closed_loop',  # closed_loop | oracle
    'aniso_closed_loop_alpha_prior': 0.0,
    'aniso_closed_loop_conf_thresh': 0.60,
    'test_init_bs': 32,
    'test_facies_bs': 64,
    'aniso_backend': 'grid',
    'aniso_use_tensor_strength': False,
    'aniso_tensor_strength_power': 1.0,
    'agpe_skel_p_thresh': 0.55,
    'agpe_skel_min_nodes': 30,
    'agpe_skel_snap_radius': 5,
    'agpe_long_edges': True,
    'agpe_long_max_step': 6,
    'agpe_long_step': 2,
    'agpe_long_cos_thresh': 0.70,
    'agpe_long_weight': 0.35,
    'agpe_edge_tau_p': 0.25,
    'agpe_lift_sigma': 2.2,
    'agpe_well_seed_mode': 'depth_gate',
    'agpe_well_seed_power': 1.0,
    'agpe_well_seed_min': 0.02,
    'agpe_well_seed_use_conf': True,
    'agpe_well_soft_alpha': 0.20,
    'agpe_cache_graph': True,
    'agpe_refine_graph': True,
    'agpe_rebuild_every': 50,
    'agpe_topo_change_pch_l1': 0.05,
}

