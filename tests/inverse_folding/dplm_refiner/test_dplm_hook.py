"""Signature guard for DPLM forward_decoder/generate hook.

Uses AST inspection to avoid pulling DPLM's full runtime dependencies
(omegaconf, hydra-core, etc.) into the unit-test process.
"""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DPLM_INVFOLD = (
    PROJECT_ROOT / "inverse_folding/dplm/src/byprot/models/dplm/dplm_invfold.py"
)


def _get_function_args(class_name: str, func_name: str) -> list[str]:
    tree = ast.parse(DPLM_INVFOLD.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == func_name:
                    return [a.arg for a in item.args.args] + [a.arg for a in item.args.kwonlyargs]
    raise AssertionError(f"{class_name}.{func_name} not found in {DPLM_INVFOLD}")


def test_forward_decoder_accepts_logit_processor_and_batch():
    args = _get_function_args("DPLMInvFold", "forward_decoder")
    assert "logit_processor" in args, args
    assert "batch" in args, args


def test_generate_accepts_logit_processor():
    args = _get_function_args("DPLMInvFold", "generate")
    assert "logit_processor" in args, args


def test_generate_reparam_decoding_keeps_current_decoder_predictions():
    """The inverse-folding generate path must match DPLM's reparam contract.

    ``output_tokens`` is the previous state being updated; ``cur_tokens`` is
    the decoder prediction from the current step. Reversing these leaves final
    residue positions as ``<mask>`` at the last step.
    """
    tree = ast.parse(DPLM_INVFOLD.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "_reparam_decoding":
            kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            if "decoding_strategy" in kwargs:
                assert kwargs["output_tokens"] == 'prev_decoder_out["output_tokens"].clone()'
                assert kwargs["output_scores"] == 'prev_decoder_out["output_scores"].clone()'
                assert kwargs["cur_tokens"] == "output_tokens.clone()"
                assert kwargs["cur_scores"] == "output_scores.clone()"
                return
    raise AssertionError("_reparam_decoding call not found in DPLMInvFold.generate")
