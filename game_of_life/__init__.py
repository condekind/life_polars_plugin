import fileinput
from collections import OrderedDict
from itertools import tee, islice
from os import PathLike

from pathlib import Path
from typing import Iterable, Any

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

    padding_len = len(str(len(board) - 1))
    board_t_dict = {f"{idx:0{padding_len}}": row for idx, row in enumerate(board)}
    return pl.DataFrame(
        board_t_dict,
    )


def _nwise_wrapping(iterable: Iterable[Any], n: int):
    """
    Returns overlapping n-tuples from an iterable, wrapping around. This means
    the result will have the same length as `iterable`. It also  means the first
    element(s) will include elements from the end of the iterable, and
    likewise, the last element(s) will include elements from the start, e.g.::

    fn('ABCDE', 3) -> 'EAB', 'ABC', 'BCD', 'CDE', 'DEA'
    """
    elements = list(iterable)
    to_be_wrapped = elements[-(n - 2) :] + elements + elements[: n - 2]
    iterators = tee(to_be_wrapped, n)
    return [
        list(z) for z in zip(*(islice(it, i, None) for i, it in enumerate(iterators)))
    ]


def step(df: pl.DataFrame, n: int = 1):
    """
    Takes a df and returns df.with_columns(...) corresponding to `n` advanced
    steps in the simulation
    """
    padding_len = len(str(df.width - 1))

    # colnums: [['{n-1}', '00', '01'], ['00', '01', '02'], ['01', '02', '03'], ... ]
    colnums = _nwise_wrapping([f"{idx:0{padding_len}}" for idx in range(df.width)], 3)

    # colnames: ['00', '01', '02', '03', ... , '{n-1}']
    colnames = [cols[1] for cols in colnums]

    # colvalues: [<Expr ['col("00")./home/…'] at 0x7B7C253C7E60>, ... ]
    colvalues = [life_step(*tuple(cols)) for cols in colnums]

    for _ in range(n):
        df = df.with_columns(**OrderedDict(zip(colnames, colvalues)))
    return df


def life_step(left: IntoExpr, mid: IntoExpr, right: IntoExpr) -> pl.Expr:
    """
    This is the function that registers the polars plugin. To use it directly,
    data must be in the correct format. An interesting way to do so is to use
    the same column names as the original data frame, so the resulting df will
    have the same shape. See how this is done in the `step(df, n)` function.
    """
    return register_plugin(
        args=[left, mid, right],
        lib=lib,
        symbol="life_step",
        is_elementwise=False,
    )
