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
