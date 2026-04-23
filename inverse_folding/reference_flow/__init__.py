"""Phase C reference-flow sampling utilities."""

from .amplification import amplification_factor, normalize_h_values, shuffle_h_values
from .config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    ReferenceFlowConfigError,
    SamplerConfig,
    ScheduleConfig,
    load_reference_flow_config,
    reference_flow_config_to_dict,
    with_reference_flow_overrides,
)
from .sampler import PositionDependentDFMSampler, SamplerOutput

__all__ = [
    "AmplificationConfig",
    "HShuffleConfig",
    "PositionDependentDFMSampler",
    "ReferenceFlowConfig",
    "ReferenceFlowConfigError",
    "SamplerConfig",
    "SamplerOutput",
    "ScheduleConfig",
    "amplification_factor",
    "load_reference_flow_config",
    "normalize_h_values",
    "reference_flow_config_to_dict",
    "shuffle_h_values",
    "with_reference_flow_overrides",
]
