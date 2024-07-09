from __future__ import annotations
import fileinput
from os import PathLike

from pathlib import Path
from typing import Iterable

import polars as pl

from game_of_life.utils import register_plugin

from polars._typing import IntoExpr


lib = Path(__file__).parent


def parse_board(
    ifile: str
    | bytes
    | PathLike[str]
    | PathLike[bytes]
    | Iterable[str | bytes | PathLike[str] | PathLike[bytes]],
) -> list[list[int]]:
    """
    Converts a board in a file containing only 0s and 1s, e.g.::

        0010
        0100

    into:
    [[0010],[0100]]
    """
    try:
        board = [
            [c for ch in ln.strip() if (c := int(ch)) in [0, 1]]
            for line in fileinput.input(ifile)
            if len(ln := line.strip()) > 0
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file: {str(ifile)} not found!")
    return board


def _transpose(board: list[list[int]]) -> list[list[int]]:
    return [[row[idx] for row in board] for idx in range(len(board[0]))]


def board_to_df(board: list[list[int]]) -> pl.DataFrame:
    """
    Converts a list of lists of integers (0s and 1s) to a Polars DataFrame.
    The inner lists must have the same length.
    """

    # This is done because each row will become a column - the user likely
    # expects a dataframe that matches the input file
    board = _transpose(board)

    padding_len = len(str(len(board)))
    board_t_dict = {f"{idx:0{padding_len}}": row for idx, row in enumerate(board)}
    return pl.DataFrame(
        board_t_dict,
    )


def life_step(left: IntoExpr, mid: IntoExpr, right: IntoExpr) -> pl.Expr:
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="life_step",
        is_elementwise=False,
    )
