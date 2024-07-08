from __future__ import annotations

from pathlib import Path

import polars as pl

from game_of_life.utils import register_plugin

from polars._typing import IntoExpr


lib = Path(__file__).parent

def life_step(left: IntoExpr, mid: IntoExpr, right: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="life_step",
        is_elementwise=False,
    )