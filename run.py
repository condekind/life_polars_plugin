import argparse
import contextlib
import io
import sys
from collections import OrderedDict
from itertools import tee, islice
from time import sleep
from typing import Iterable, Any

from game_of_life import life_step, parse_board, board_to_df
import polars as pl


def nwise(iterable: Iterable[Any], n: int):
    """Return overlapping n-tuples from an iterable."""
    iterators = tee(iterable, n)
    return [
        list(z) for z in zip(*(islice(it, i, None) for i, it in enumerate(iterators)))
    ]


def nwise_wrapping(iterable: Iterable[Any], n: int):
    """Return overlapping n-tuples from an iterable."""
    elements = list(iterable)
    to_be_wrapped = elements[-(n - 2) :] + elements + elements[: n - 2]
    iterators = tee(to_be_wrapped, n)
    return [
        list(z) for z in zip(*(islice(it, i, None) for i, it in enumerate(iterators)))
    ]


class Application:
    def __init__(self):
        self._args = argparse.Namespace()
        cli = argparse.ArgumentParser(
            prog="python -m game_of_life", description="Options"
        )
        cli.add_argument("-i", "--input", type=str, required=True)
        cli.add_argument("-d", "--delay", type=float, default=0.2)
        cli.add_argument("-n", "--num-steps", type=int, default=sys.maxsize)
        # Not impl
        # cli.add_argument('--alive-char', type=str, required='█')
        # cli.add_argument('--dead-char', type=str, required='░')
        cli.parse_args(namespace=self._args)

        # [-i]
        self.ifile: str = self._args.input

        # [-d]
        self.delay: float = self._args.delay

        # [-n]
        self.steps: int = self._args.num_steps

        # Creates a pl.DataFrame from the provided file
        self.df = board_to_df(parse_board(self.ifile))

    def __str__(self) -> str:
        res = io.StringIO()
        with (
            pl.Config(tbl_rows=-1, tbl_cols=-1),
            contextlib.redirect_stdout(res),
        ):
            print(self.df)
        return res.getvalue()

    def start(
        self,
        n: int | None = None,
        delay: float | None = None,
        print_df: bool = True,
    ):
        if n is None:
            n = self.steps

        if delay is None:
            delay = self.delay

        # colnums: [['00', '01', '02'], ['01', '02', '03'], ... ]
        colnums = nwise_wrapping([f"{idx:02}" for idx in range(self.df.width)], 3)

        # colnames: ['01', '02', '03', ... ]
        colnames = [cols[1] for cols in colnums]

        # colvalues: [<Expr ['col("00")./home/…'] at 0x7B7C253C7E60>, ... ]
        colvalues = [life_step(*tuple(cols)) for cols in colnums]

        if n <= 1 and print_df:
            print(self)
            return

        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            cnt = 1
            try:
                for _ in range(n):
                    self.df = self.df.with_columns(
                        **OrderedDict(zip(colnames, colvalues)),
                    )
                    # Clear screen
                    print("\033[2J")
                    print(self.df)
                    cnt += 1
                    sleep(delay)
            except KeyboardInterrupt:
                print(f"\nKeyboard Interrupt: ran for {cnt} iterations. Aborting...")
                print(f"{self._args.num_steps=}\n{self._args.delay}")


if __name__ == "__main__":
    app = Application()
    app.start()
