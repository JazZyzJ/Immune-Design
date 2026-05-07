"""Contract tests for the wandb helper — must no-op safely without wandb installed."""

from __future__ import annotations

import argparse

import pytest

from inverse_folding.observability import (
    add_wandb_cli_args,
    finish_wandb,
    init_wandb_from_args,
    is_active,
    log_metrics,
    set_summary,
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_wandb_cli_args(parser, default_project="test-project")
    return parser.parse_args(argv)


def test_default_args_disable_wandb() -> None:
    args = _parse([])
    assert args.wandb is False
    assert args.wandb_project == "test-project"
    assert args.wandb_mode == "online"


def test_init_returns_none_when_disabled() -> None:
    args = _parse([])
    run = init_wandb_from_args(args, run_name="r0", config={"k": 1})
    assert run is None
    assert is_active(run) is False


def test_log_and_summary_are_noops_when_disabled() -> None:
    log_metrics(None, {"foo": 1.0})
    set_summary(None, {"bar": "value"})
    finish_wandb(None)


def test_init_returns_none_when_mode_disabled_even_with_flag() -> None:
    args = _parse(["--wandb", "--wandb-mode", "disabled"])
    run = init_wandb_from_args(args, run_name="r0", config={})
    assert run is None


def test_wandb_flag_with_no_package_falls_back_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the helper into the unavailable branch even if wandb is installed locally.
    import inverse_folding.observability.wandb_runtime as runtime

    monkeypatch.setattr(runtime, "_AVAILABLE", False)
    monkeypatch.setattr(runtime, "_wandb", None)
    args = _parse(["--wandb"])
    run = init_wandb_from_args(args, run_name="r0", config={})
    assert run is None
