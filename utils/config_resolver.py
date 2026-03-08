from __future__ import annotations

import copy
from typing import Any, Mapping


DEFAULT_SEED = 2026
DEFAULT_SELECTED_WELLS_CSV = r"data/Stanford_VI/selected_wells_20_seed2026.csv"
DETAIL_HF_RATIO = 0.20


TRAIN_USER_ALLOWED_KEYS = frozenset(
    {
        "batch_size",
        "no_wells",
        "lambda_recon",
        "lambda_facies",
        "lambda_detail",
        "use_boundary_weight",
        "boundary_beta",
        "epochs",
        "grad_clip",
        "lr",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "seed",
        "selected_wells_csv",
        "use_aniso_conditioning",
        "R_update_every",
        "use_detail_branch",
        "detail_gain",
        "channel_id",
        "model_name",
        "Forward_model",
        "Facies_model",
        "run_id_suffix",
        "data_flag",
    }
)

TEST_USER_ALLOWED_KEYS = frozenset(
    {
        "no_wells",
        "seed",
        "selected_wells_csv",
        "data_flag",
        "use_aniso_conditioning",
        "test_init_bs",
        "test_facies_bs",
        "channel_id",
        "run_id",
        "model_name",
        "run_id_suffix",
        "test_noise_kind",
        "test_noise_snr_db",
        "test_noise_seed",
        "test_noise_save_inputs",
    }
)


_TRAIN_BASE_DEFAULTS: dict[str, Any] = {
    "batch_size": 4,
    "no_wells": 20,
    "unsupervised_seismic": -1,
    "lambda_ai": 5.0,
    "lambda_facies": 0.2,
    "lambda_recon": 1.0,
    "use_two_stage_loss": True,
    "stageA_epochs": 200,
    "stageB_ramp_epochs": 200,
    "stageA_lambda_ai_mult": 1.8,
    "stageA_lambda_facies_mult": 0.30,
    "stageA_lambda_recon_mult": 0.30,
    "lambda_amp_anchor": 0.05,
    "lambda_depth_grad": 0.20,
    "lambda_depth_hf": 0.10,
    "depth_detail_norm": "l1",
    "hf_second_order": True,
    "use_depth_warm_schedule": False,
    "depth_warm_start_epoch": 0,
    "depth_warm_ramp_epochs": 1,
    "use_boundary_weight": True,
    "boundary_weight_beta": 1.5,
    "boundary_weight_width": 2,
    "boundary_weight_max": 4.0,
    "use_boundary_warm_schedule": False,
    "boundary_beta_start": 1.5,
    "boundary_beta_end": 1.5,
    "boundary_warm_start_epoch": 0,
    "boundary_warm_ramp_epochs": 1,
    "boundary_weight_apply_ai": True,
    "boundary_weight_apply_detail": True,
    "boundary_weight_apply_facies": False,
    "facies_detach_y": False,
    "use_late_multitask_decouple": True,
    "late_multitask_start_epoch": 450,
    "facies_detach_y_late": True,
    "epochs": 1000,
    "grad_clip": 1.0,
    "lr": 0.0001,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "train_noise_kind": "none",
    "train_noise_prob": 0.0,
    "train_noise_snr_db_choices": (),
    "r_channel_dropout_prob": 0.0,
    "ws_every": 2,
    "ws_max_batches": 150,
    "ws_max_batches_stageA": 50,
    "ws_every_late": 10,
    "ws_max_batches_late": 15,
    "seed": DEFAULT_SEED,
    "selected_wells_csv": DEFAULT_SELECTED_WELLS_CSV,
    "use_aniso_conditioning": True,
    "aniso_train_protocol": "closed_loop",
    "aniso_closed_loop_alpha_prior": 0.0,
    "aniso_closed_loop_conf_thresh": 0.60,
    "agpe_neutral_p": 0.5,
    "channel_id": 2,
    "aniso_steps_R": 25,
    "aniso_eta": 0.6,
    "aniso_gamma": 8.0,
    "aniso_tau": 0.6,
    "aniso_kappa": 4.0,
    "aniso_sigma_st": 1.2,
    "aniso_backend": "grid",
    "aniso_use_tensor_strength": False,
    "aniso_tensor_strength_power": 1.0,
    "agpe_skel_p_thresh": 0.55,
    "agpe_skel_min_nodes": 30,
    "agpe_skel_snap_radius": 5,
    "agpe_long_edges": True,
    "agpe_long_max_step": 6,
    "agpe_long_step": 2,
    "agpe_long_cos_thresh": 0.70,
    "agpe_long_weight": 0.35,
    "agpe_edge_tau_p": 0.25,
    "agpe_lift_sigma": 2.2,
    "agpe_well_seed_mode": "depth_gate",
    "agpe_well_seed_power": 1.0,
    "agpe_well_seed_min": 0.02,
    "agpe_well_seed_use_conf": True,
    "agpe_well_soft_alpha": 0.20,
    "agpe_cache_graph": True,
    "agpe_refine_graph": True,
    "agpe_rebuild_every": 50,
    "agpe_topo_change_pch_l1": 0.05,
    "use_soft_prior": False,
    "aniso_steps_prior": 35,
    "lambda_prior": 0.20,
    "iterative_R": True,
    "R_update_every": 50,
    "R_update_bs": 16,
    "R_ema_beta": 0.85,
    "alpha_prior_start": 1.0,
    "alpha_prior_end": 0.3,
    "alpha_prior_decay_epochs": 800,
    "conf_thresh": 0.75,
    "lambda_phys_damp": 0.8,
    "save_R_every": 50,
    "use_detail_branch": True,
    "detail_gain": 0.15,
    "detail_hp_kernel": 9,
    "detail_channels": 24,
    "detail_dilations": (1, 2, 4),
    "detail_kernel_sizes": (9, 7, 5),
    "model_name": "VishalNet",
    "Forward_model": "cov_para",
    "Facies_model": "Facies",
    "run_id_suffix": "",
    "data_flag": "Stanford_VI",
}

_TEST_BASE_DEFAULTS: dict[str, Any] = {
    "no_wells": 20,
    "seed": DEFAULT_SEED,
    "selected_wells_csv": DEFAULT_SELECTED_WELLS_CSV,
    "data_flag": "Stanford_VI",
    "model_name": "VishalNet_cov_para_Facies_s_uns",
    "run_id": "VishalNet_cov_para_Facies",
    "run_id_suffix": "",
    "use_aniso_conditioning": True,
    "aniso_test_protocol": "closed_loop",
    "aniso_closed_loop_alpha_prior": 0.0,
    "aniso_closed_loop_conf_thresh": 0.60,
    "test_init_bs": 32,
    "test_facies_bs": 64,
    "channel_id": 2,
    "aniso_backend": "grid",
    "aniso_use_tensor_strength": False,
    "aniso_tensor_strength_power": 1.0,
    "agpe_skel_p_thresh": 0.55,
    "agpe_skel_min_nodes": 30,
    "agpe_skel_snap_radius": 5,
    "agpe_long_edges": True,
    "agpe_long_max_step": 6,
    "agpe_long_step": 2,
    "agpe_long_cos_thresh": 0.70,
    "agpe_long_weight": 0.35,
    "agpe_edge_tau_p": 0.25,
    "agpe_lift_sigma": 2.2,
    "agpe_well_seed_mode": "depth_gate",
    "agpe_well_seed_power": 1.0,
    "agpe_well_seed_min": 0.02,
    "agpe_well_seed_use_conf": True,
    "agpe_well_soft_alpha": 0.20,
    "agpe_cache_graph": True,
    "agpe_refine_graph": True,
    "agpe_rebuild_every": 50,
    "agpe_topo_change_pch_l1": 0.05,
    "test_noise_kind": "none",
    "test_noise_snr_db": None,
    "test_noise_seed": DEFAULT_SEED,
    "test_noise_save_inputs": True,
}


_TRAIN_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "legacy_grid": {},
    "stable_r51": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 100,
        "R_update_every": 50,
        "aniso_closed_loop_conf_thresh": 0.60,
        "agpe_lift_sigma": 2.3,
        "agpe_long_weight": 0.35,
        "agpe_well_soft_alpha": 0.20,
        "agpe_well_seed_mode": "depth_gate",
        "agpe_well_seed_min": 0.02,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.005,
        "lambda_depth_hf": 0.001,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 500,
        "depth_warm_ramp_epochs": 300,
        "use_boundary_warm_schedule": False,
        "ws_every": 4,
        "ws_max_batches": 80,
        "ws_max_batches_stageA": 30,
        "ws_every_late": 12,
        "ws_max_batches_late": 8,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.05,
        "boundary_weight_width": 1,
        "boundary_weight_apply_ai": True,
        "boundary_weight_apply_detail": False,
        "boundary_weight_apply_facies": False,
    },
}

_TEST_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "legacy_grid": {},
    "stable_r51": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "aniso_closed_loop_conf_thresh": 0.60,
        "agpe_long_edges": True,
        "agpe_lift_sigma": 2.3,
        "agpe_well_soft_alpha": 0.20,
        "agpe_well_seed_mode": "depth_gate",
        "agpe_well_seed_min": 0.02,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 100,
    },
}


_BOUNDARY_DISABLED_PRESET: dict[str, Any] = {
    "use_boundary_weight": False,
    "boundary_weight_beta": 0.15,
    "boundary_weight_width": 1,
    "use_boundary_warm_schedule": False,
    "boundary_beta_start": 0.0,
    "boundary_beta_end": 0.15,
    "boundary_warm_start_epoch": 500,
    "boundary_warm_ramp_epochs": 300,
    "boundary_weight_apply_ai": True,
    "boundary_weight_apply_detail": False,
    "boundary_weight_apply_facies": False,
}

_BOUNDARY_AI_ONLY_LATE_WEAK_PRESET: dict[str, Any] = {
    "use_boundary_weight": True,
    "boundary_weight_width": 1,
    "use_boundary_warm_schedule": True,
    "boundary_warm_start_epoch": 500,
    "boundary_warm_ramp_epochs": 300,
    "boundary_weight_apply_ai": True,
    "boundary_weight_apply_detail": False,
    "boundary_weight_apply_facies": False,
}


_TRAIN_DIRECT_USER_KEYS = (
    "batch_size",
    "no_wells",
    "lambda_recon",
    "lambda_facies",
    "epochs",
    "grad_clip",
    "lr",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "seed",
    "selected_wells_csv",
    "use_aniso_conditioning",
    "R_update_every",
    "use_detail_branch",
    "detail_gain",
    "channel_id",
    "model_name",
    "Forward_model",
    "Facies_model",
    "run_id_suffix",
    "data_flag",
)

_TEST_DIRECT_USER_KEYS = (
    "no_wells",
    "seed",
    "selected_wells_csv",
    "data_flag",
    "use_aniso_conditioning",
    "test_init_bs",
    "test_facies_bs",
    "channel_id",
    "run_id",
    "model_name",
    "run_id_suffix",
    "test_noise_kind",
    "test_noise_snr_db",
    "test_noise_seed",
    "test_noise_save_inputs",
)

_TEST_SHARED_FROM_TRAIN_KEYS = (
    "no_wells",
    "seed",
    "selected_wells_csv",
    "data_flag",
    "use_aniso_conditioning",
    "channel_id",
    "aniso_backend",
    "aniso_use_tensor_strength",
    "aniso_tensor_strength_power",
    "agpe_skel_p_thresh",
    "agpe_skel_min_nodes",
    "agpe_skel_snap_radius",
    "agpe_long_edges",
    "agpe_long_max_step",
    "agpe_long_step",
    "agpe_long_cos_thresh",
    "agpe_long_weight",
    "agpe_edge_tau_p",
    "agpe_lift_sigma",
    "agpe_well_seed_mode",
    "agpe_well_seed_power",
    "agpe_well_seed_min",
    "agpe_well_seed_use_conf",
    "agpe_well_soft_alpha",
    "agpe_cache_graph",
    "agpe_refine_graph",
    "agpe_rebuild_every",
    "agpe_topo_change_pch_l1",
    "aniso_closed_loop_alpha_prior",
    "aniso_closed_loop_conf_thresh",
)


def _to_dict(data: Mapping[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    return {str(k): copy.deepcopy(v) for k, v in data.items()}


def _validate_keys(kind: str, cfg: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(cfg.keys()) - set(allowed))
    if unknown:
        raise KeyError(f"Unknown {kind} config keys: {unknown}")


def _merge_dicts(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        if part is None:
            continue
        out.update(_to_dict(part))
    return out


def _apply_boundary_controls(cfg: dict[str, Any], enabled: bool, beta: float) -> None:
    beta_value = float(beta)
    if enabled:
        cfg.update(copy.deepcopy(_BOUNDARY_AI_ONLY_LATE_WEAK_PRESET))
        cfg["boundary_weight_beta"] = beta_value
        cfg["boundary_beta_start"] = 0.0
        cfg["boundary_beta_end"] = beta_value
        return

    cfg.update(copy.deepcopy(_BOUNDARY_DISABLED_PRESET))
    cfg["boundary_weight_beta"] = beta_value
    cfg["boundary_beta_start"] = 0.0
    cfg["boundary_beta_end"] = beta_value


def _apply_detail_controls(cfg: dict[str, Any], lambda_detail: float | None) -> None:
    if lambda_detail is None:
        return

    detail = max(float(lambda_detail), 0.0)
    cfg["lambda_depth_grad"] = detail
    cfg["lambda_depth_hf"] = detail * DETAIL_HF_RATIO
    cfg["use_depth_warm_schedule"] = detail > 0.0
    if detail <= 0.0:
        cfg["depth_warm_start_epoch"] = 0
        cfg["depth_warm_ramp_epochs"] = 1


def build_train_config(
    profile: str = "stable_r51",
    user_cfg: Mapping[str, Any] | None = None,
    expert_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if profile not in _TRAIN_PROFILE_PRESETS:
        raise KeyError(f"Unknown train profile: {profile}")

    user = _to_dict(user_cfg)
    expert = _to_dict(expert_overrides)
    _validate_keys("train user", user, TRAIN_USER_ALLOWED_KEYS)

    cfg = _merge_dicts(_TRAIN_BASE_DEFAULTS, _TRAIN_PROFILE_PRESETS[profile])

    for key in _TRAIN_DIRECT_USER_KEYS:
        if key in user:
            cfg[key] = copy.deepcopy(user[key])

    if "lambda_detail" in user:
        _apply_detail_controls(cfg, float(user["lambda_detail"]))

    boundary_enabled = bool(user.get("use_boundary_weight", cfg.get("use_boundary_weight", False)))
    boundary_beta = float(user.get("boundary_beta", cfg.get("boundary_weight_beta", 0.15)))
    _apply_boundary_controls(cfg, enabled=boundary_enabled, beta=boundary_beta)

    if not bool(cfg.get("use_aniso_conditioning", True)):
        cfg["iterative_R"] = False

    cfg.update(expert)
    return cfg


def build_test_config(
    profile: str = "stable_r51",
    user_cfg: Mapping[str, Any] | None = None,
    expert_overrides: Mapping[str, Any] | None = None,
    train_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if profile not in _TEST_PROFILE_PRESETS:
        raise KeyError(f"Unknown test profile: {profile}")

    user = _to_dict(user_cfg)
    expert = _to_dict(expert_overrides)
    _validate_keys("test user", user, TEST_USER_ALLOWED_KEYS)

    cfg = _merge_dicts(_TEST_BASE_DEFAULTS, _TEST_PROFILE_PRESETS[profile])

    if train_cfg is not None:
        for key in _TEST_SHARED_FROM_TRAIN_KEYS:
            if key in train_cfg:
                cfg[key] = copy.deepcopy(train_cfg[key])

        if "run_id_suffix" not in user and "run_id_suffix" not in expert:
            cfg["run_id_suffix"] = copy.deepcopy(train_cfg.get("run_id_suffix", ""))
        if "run_id" not in user and "run_id" not in expert:
            run_id_base = (
                f"{train_cfg['model_name']}_{train_cfg['Forward_model']}_{train_cfg['Facies_model']}"
            )
            cfg["run_id"] = run_id_base
        if "model_name" not in user and "model_name" not in expert:
            cfg["model_name"] = f"{cfg['run_id']}_s_uns"

    for key in _TEST_DIRECT_USER_KEYS:
        if key in user:
            cfg[key] = copy.deepcopy(user[key])

    if not bool(cfg.get("use_aniso_conditioning", True)):
        cfg["aniso_backend"] = copy.deepcopy(_TEST_BASE_DEFAULTS["aniso_backend"])
        cfg["aniso_use_tensor_strength"] = copy.deepcopy(_TEST_BASE_DEFAULTS["aniso_use_tensor_strength"])

    cfg.update(expert)
    return cfg


def resolve_train_config(
    cfg: Mapping[str, Any] | None = None,
    *,
    default_profile: str = "stable_r51",
) -> dict[str, Any]:
    raw = _to_dict(cfg)
    profile = str(raw.pop("profile", default_profile))
    nested_expert = raw.pop("expert_overrides", None)

    user_cfg = {k: raw.pop(k) for k in list(raw.keys()) if k in TRAIN_USER_ALLOWED_KEYS}
    expert_cfg = _merge_dicts(raw, nested_expert)

    return build_train_config(
        profile=profile,
        user_cfg=user_cfg,
        expert_overrides=expert_cfg,
    )


def resolve_test_config(
    cfg: Mapping[str, Any] | None = None,
    *,
    train_cfg: Mapping[str, Any] | None = None,
    default_profile: str = "stable_r51",
    default_train_profile: str | None = None,
) -> dict[str, Any]:
    raw = _to_dict(cfg)
    profile = str(raw.pop("profile", default_profile))
    nested_expert = raw.pop("expert_overrides", None)

    user_cfg = {k: raw.pop(k) for k in list(raw.keys()) if k in TEST_USER_ALLOWED_KEYS}
    expert_cfg = _merge_dicts(raw, nested_expert)

    resolved_train = None
    if train_cfg is not None:
        resolved_train = resolve_train_config(
            train_cfg,
            default_profile=default_train_profile or default_profile,
        )

    return build_test_config(
        profile=profile,
        user_cfg=user_cfg,
        expert_overrides=expert_cfg,
        train_cfg=resolved_train,
    )
