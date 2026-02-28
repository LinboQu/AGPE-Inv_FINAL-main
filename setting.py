import numpy as np 
from scipy.spatial import distance

# -----------------------------
# Selected wells (seed2026)
# -----------------------------
SEED = 2026
SELECTED_WELLS_CSV = r"data/Stanford_VI/selected_wells_20_seed2026.csv"

TCN1D_train_p = {'batch_size': 4,
				'no_wells': 401, 
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
                # DataLoader acceleration
				'num_workers': 4,             # Windows建议 2~8；先用4
				'pin_memory': True,           # GPU训练建议True
				'persistent_workers': True,   # num_workers>0 时有效
                # Weak-supervision scheduling 
				'ws_every': 2,                # 每5个epoch跑一次弱监督（大幅提速）
                'ws_max_batches': 150,         # 每次最多跑50个弱监督batch（再提速）
                "seed": SEED,
                "no_wells": 20,                   # ✅ 用真实 20 口井
                "selected_wells_csv": SELECTED_WELLS_CSV,   # ✅ 井位种子来源

				# -----------------------------
				# Facies-adaptive anisotropic conditioning (FARP)
				# A reliability field R(x) is propagated from sparse wells in a facies- and
				# waveform-similarity induced anisotropic metric space, then injected as an
				# extra input channel (and optional soft prior loss).
				# NOTE: default is False to keep original GPI behavior.
				'use_aniso_conditioning': True,
				'channel_id': 2,            # VI-E: 2 is Channel
				'aniso_steps_R': 25,
				'aniso_eta': 0.6,
				'aniso_gamma': 8.0,
				'aniso_tau': 0.6,
				'aniso_kappa': 4.0,
				'aniso_sigma_st': 1.2,
				# optional soft impedance prior (propagated from wells)
				'use_soft_prior': False,
				'aniso_steps_prior': 35,
				'lambda_prior': 0.20,
				# -----------------------------
				# Iterative coupling for R(x) (optional)
				# Rebuild R from predicted facies + physics residual every few epochs,
				# and EMA-update to avoid drift.
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


    # SOFT模型：1、VishalNet, 2、GRU_MM, 3、Unet_1D, 4、Unet_1D_convolution
    # 反演模型：Transformer_cov_para_geo：Transformer+cov_para+geo
    # 消融实验模型： 1、Tansformer_cov_para, 2、Tansformer_geo, 3、Tansformer_convolution_geo

				'model_name': 'VishalNet',
				'Forward_model': 'cov_para',       # '' 表示没有正演过程
				'Facies_model': 'Facies',

	# 'Stanford_VI', 'Fanny', 
				'data_flag': 'Stanford_VI',
				'get_F': 0,  #（0,2,4） 地震数据扩充了频率特征和动态特征，当0时表示只有时域地震波形
				'F': 'WE_PreSDM',  # 当："data_flag = M2_F
				}

TCN1D_test_p = {"no_wells": 20,                   # ✅ 用真实 20 口井
                "seed": SEED,
                "selected_wells_csv": SELECTED_WELLS_CSV,   # ✅ 井位种子来源               
				'data_flag':'Stanford_VI',
				'model_name': 'VishalNet_cov_para_Facies_s_uns',  # model-name_Forward-model_Facies-model_s_uns
                "run_id": "VishalNet_cov_para_Facies",   # ✅ 用于找 full_ckpt/stats
				# 注意，如果是有正演模块，则：反演模块_正演模块
				# If you trained with anisotropic conditioning, enable it in test as well.
				'use_aniso_conditioning': True,
				}

