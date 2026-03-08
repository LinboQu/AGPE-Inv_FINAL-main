from utils.config_resolver import (
    DEFAULT_SEED,
    DEFAULT_SELECTED_WELLS_CSV,
    build_test_config,
    build_train_config,
)


SEED = DEFAULT_SEED
SELECTED_WELLS_CSV = DEFAULT_SELECTED_WELLS_CSV

# Stable profiles:
# - "stable_r51": current recommended default
# - "legacy_grid": legacy pre-R51 baseline
TRAIN_PROFILE = "stable_r51"
TEST_PROFILE = TRAIN_PROFILE

# Day-to-day knobs only. Internal schedules and AGPE topology details live in the profile.
# lambda_detail expands to:
#   lambda_depth_grad = lambda_detail
#   lambda_depth_hf   = 0.2 * lambda_detail
# If use_boundary_weight=True, the resolver applies a fixed AI-only late/weak boundary mode.
TRAIN_USER_P = {
    "batch_size": 4,
    "no_wells": 20,
    "lambda_recon": 1.0,
    "lambda_facies": 0.2,
    "lambda_detail": 0.005,
    "use_boundary_weight": False,
    "boundary_beta": 0.15,
    "epochs": 1000,
    "grad_clip": 1.0,
    "lr": 0.0001,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": SEED,
    "selected_wells_csv": SELECTED_WELLS_CSV,
    "use_aniso_conditioning": True,
    "R_update_every": 50,
    "use_detail_branch": True,
    "detail_gain": 0.15,
    "channel_id": 2,
    "model_name": "VishalNet",
    "Forward_model": "cov_para",
    "Facies_model": "Facies",
    "run_id_suffix": "",
    "data_flag": "Stanford_VI",
}

# Expert overrides bypass the compact user layer and write raw internal keys.
TRAIN_EXPERT_OVERRIDES = {}

TCN1D_train_p = build_train_config(
    profile=TRAIN_PROFILE,
    user_cfg=TRAIN_USER_P,
    expert_overrides=TRAIN_EXPERT_OVERRIDES,
)

TEST_USER_P = {
    "no_wells": TRAIN_USER_P["no_wells"],
    "seed": TRAIN_USER_P["seed"],
    "selected_wells_csv": TRAIN_USER_P["selected_wells_csv"],
    "data_flag": TRAIN_USER_P["data_flag"],
    "use_aniso_conditioning": TRAIN_USER_P["use_aniso_conditioning"],
    "channel_id": TRAIN_USER_P["channel_id"],
    "test_init_bs": 32,
    "test_facies_bs": 64,
    "test_noise_kind": "none",
    "test_noise_snr_db": None,
    "test_noise_seed": SEED,
    "test_noise_save_inputs": True,
}

TEST_EXPERT_OVERRIDES = {}

TCN1D_test_p = build_test_config(
    profile=TEST_PROFILE,
    user_cfg=TEST_USER_P,
    expert_overrides=TEST_EXPERT_OVERRIDES,
    train_cfg=TCN1D_train_p,
)

