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
    'seed': SEED,
    'selected_wells_csv': SELECTED_WELLS_CSV,

    # Facies-adaptive anisotropic conditioning
    'use_aniso_conditioning': True,
    'channel_id': 2,
    'aniso_steps_R': 25,
    'aniso_eta': 0.6,
    'aniso_gamma': 8.0,
    'aniso_tau': 0.6,
    'aniso_kappa': 4.0,
    'aniso_sigma_st': 1.2,
    'aniso_backend': 'grid',                # grid | graph_lattice
    # default False for parity: graph_lattice ~= grid
    'aniso_use_tensor_strength': False,
    'aniso_tensor_strength_power': 1.0,

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
    'aniso_backend': 'grid',
    'aniso_use_tensor_strength': False,
    'aniso_tensor_strength_power': 1.0,
}

