"""Config loader for Epitope Head."""

from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent


def load_data_config(path: Path | str | None = None) -> dict:
    """Load and validate data config. Raises on missing required keys."""
    if path is None:
        path = _CONFIG_DIR / "data.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data")
    if data is None:
        raise ValueError("Config missing top-level 'data' key")

    required = ["target_allele", "min_k", "max_k", "coord_mode"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if not isinstance(data["target_allele"], str) or len(data["target_allele"]) == 0:
        raise ValueError("target_allele must be a non-empty string")

    return data


def load_model_config(path: Path | str | None = None) -> dict:
    """Load and validate model config."""
    if path is None:
        path = _CONFIG_DIR / "model.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model")
    if model is None:
        raise ValueError("Config missing top-level 'model' key")

    required = ["encoder_name", "freeze_encoder", "d_proj",
                 "scorer_hidden_dim", "scorer_activation"]
    missing = [k for k in required if k not in model]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    return model


def load_train_config(path: Path | str | None = None) -> dict:
    """Load and validate training config.

    Validates all E0-frozen keys including nested loss.* and chunking.* sub-configs.
    Raises ValueError on any missing or mistyped key.
    """
    if path is None:
        path = _CONFIG_DIR / "train.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    train = cfg.get("train")
    if train is None:
        raise ValueError("Config missing top-level 'train' key")

    # ── Top-level required keys (E0 frozen) ──
    required_top = [
        # Negative sampling
        "neg_ratio", "hard_negative_fraction", "hard_neg_max_overlap_ratio",
        "hard_neg_offset_range", "neg_length_sampling",
        # Loss (sub-dict validated below)
        "loss",
        # Optimization
        "lr", "weight_decay", "optimizer", "scheduler",
        "warmup_steps", "max_epochs", "grad_clip",
        # Batching
        "max_tokens", "num_workers",
        # Reproducibility
        "seed", "deterministic",
        # Checkpointing
        "checkpoint_every_n_epochs", "early_stopping_patience", "monitor_metric",
        # Chunking (sub-dict validated below)
        "chunking",
    ]
    missing = [k for k in required_top if k not in train]
    if missing:
        raise ValueError(f"Train config missing required keys: {missing}")

    # ── loss.* sub-keys ──
    loss = train["loss"]
    if not isinstance(loss, dict):
        raise ValueError("train.loss must be a dict")
    required_loss = ["tau", "T_mp", "lambda_mp", "lambda_smooth"]
    missing_loss = [k for k in required_loss if k not in loss]
    if missing_loss:
        raise ValueError(f"train.loss missing required keys: {missing_loss}")

    # ── chunking.* sub-keys ──
    chunking = train["chunking"]
    if not isinstance(chunking, dict):
        raise ValueError("train.chunking must be a dict")
    required_chunking = [
        "enabled", "context_len", "stride", "margin",
        "stitch_mode", "window_mode", "enable_reliability",
    ]
    missing_chunking = [k for k in required_chunking if k not in chunking]
    if missing_chunking:
        raise ValueError(f"train.chunking missing required keys: {missing_chunking}")

    return train


def load_ablation_config(path: Path | str | None = None) -> dict:
    """Load and validate encoder ablation config (Module H contract).

    Validates that each profile has required encoder keys and that
    frozen constants match expected values.
    """
    if path is None:
        path = _CONFIG_DIR / "model_ablation.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ablation = cfg.get("ablation")
    if ablation is None:
        raise ValueError("Config missing top-level 'ablation' key")

    profiles = ablation.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("ablation.profiles must be a non-empty dict")

    VALID_ENCODER_TYPES = {"esm2_frozen", "dilated_cnn", "shallow_transformer", "multiscale_cnn"}
    REQUIRED_PROFILE_KEYS = {"encoder_type", "d_enc", "trainable_encoder", "encoder_cfg"}

    ENCODER_CFG_KEYS = {
        "esm2_frozen": {"encoder_name"},
        "dilated_cnn": {"token_emb_dim", "n_blocks", "kernel_size", "dilations",
                        "hidden_channels", "block_dropout"},
        "shallow_transformer": {"d_model", "n_layers", "n_heads", "ffn_dim",
                                "dropout", "max_seq_len"},
        "multiscale_cnn": {"token_emb_dim", "n_blocks", "dilations",
                           "hidden_channels", "branch_channels", "block_dropout"},
    }

    OPTIONAL_PROFILE_KEYS = {"loss_overrides"}

    for profile_id, profile in profiles.items():
        missing = REQUIRED_PROFILE_KEYS - set(profile.keys())
        if missing:
            raise ValueError(
                f"Ablation profile '{profile_id}' missing required keys: {sorted(missing)}"
            )

        enc_type = profile["encoder_type"]
        if enc_type not in VALID_ENCODER_TYPES:
            raise ValueError(
                f"Profile '{profile_id}': encoder_type '{enc_type}' not in {sorted(VALID_ENCODER_TYPES)}"
            )

        enc_cfg = profile["encoder_cfg"]
        if not isinstance(enc_cfg, dict):
            raise ValueError(f"Profile '{profile_id}': encoder_cfg must be a dict")

        expected_keys = ENCODER_CFG_KEYS[enc_type]
        missing_enc = expected_keys - set(enc_cfg.keys())
        if missing_enc:
            raise ValueError(
                f"Profile '{profile_id}': encoder_cfg missing keys: {sorted(missing_enc)}"
            )

    return ablation


def load_inference_config(path: Path | str | None = None) -> dict:
    """Load and validate inference config (Module F0 contract)."""
    if path is None:
        path = _CONFIG_DIR / "inference.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    inference = cfg.get("inference")
    if inference is None:
        raise ValueError("Config missing top-level 'inference' key")

    required_top = [
        "checkpoint_path",
        "min_k",
        "max_k",
        "hotspot_center_method",
        "hotspot_clamp",
        "chunking",
        "device",
    ]
    missing = [k for k in required_top if k not in inference]
    if missing:
        raise ValueError(f"Inference config missing required keys: {missing}")

    if not isinstance(inference["min_k"], int) or not isinstance(inference["max_k"], int):
        raise ValueError("inference.min_k and inference.max_k must be integers")
    if inference["min_k"] > inference["max_k"]:
        raise ValueError("inference.min_k must be <= inference.max_k")
    if inference["min_k"] != 12:
        raise ValueError("inference.min_k must be frozen to 12 for v1.1")
    if inference["max_k"] != 25:
        raise ValueError("inference.max_k must be frozen to 25 for v1.1")

    center_allowed = {"median", "mean", "none"}
    if inference["hotspot_center_method"] not in center_allowed:
        raise ValueError(
            f"inference.hotspot_center_method must be one of {sorted(center_allowed)}",
        )

    clamp_allowed = {"softplus", "relu", "none"}
    if inference["hotspot_clamp"] not in clamp_allowed:
        raise ValueError(
            f"inference.hotspot_clamp must be one of {sorted(clamp_allowed)}",
        )

    chunking = inference["chunking"]
    if not isinstance(chunking, dict):
        raise ValueError("inference.chunking must be a dict")

    required_chunking = [
        "enabled",
        "context_len",
        "stride",
        "margin",
        "stitch_mode",
        "enable_reliability",
    ]
    missing_chunking = [k for k in required_chunking if k not in chunking]
    if missing_chunking:
        raise ValueError(f"inference.chunking missing required keys: {missing_chunking}")

    stitch_allowed = {"per_residue_stitch", "center_weighted"}
    if chunking["stitch_mode"] not in stitch_allowed:
        raise ValueError(
            f"inference.chunking.stitch_mode must be one of {sorted(stitch_allowed)}",
        )
    if not bool(chunking["enabled"]):
        raise ValueError("inference.chunking.enabled must be true in v1.1")
    if int(chunking["context_len"]) != 1022:
        raise ValueError("inference.chunking.context_len must be frozen to 1022 for v1.1")
    if int(chunking["stride"]) != 512:
        raise ValueError("inference.chunking.stride must be frozen to 512 for v1.1")
    if int(chunking["margin"]) != 32:
        raise ValueError("inference.chunking.margin must be frozen to 32 for v1.1")
    if chunking["stitch_mode"] != "per_residue_stitch":
        raise ValueError(
            "inference.chunking.stitch_mode must be frozen to per_residue_stitch in v1.1",
        )

    return inference
