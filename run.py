import argparse
import contextlib
import fileinput
import io
import sys
from itertools import tee, islice
from time import sleep
from typing import Iterable

from game_of_life import life_step
import polars as pl


def nwise[T](iterable: Iterable[T], n: int):
    """Return overlapping n-tuples from an iterable."""
    iterators = tee(iterable, n)
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

        try:
            board = [
                [c for ch in ln.strip() if (c := int(ch)) in [0, 1]]
                for line in fileinput.input(self.ifile)
                if len(ln := line.strip()) > 0
            ]
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file: '{self.ifile}' not found!")

        board_t_dict = {f"{idx:02}": row for idx, row in enumerate(board)}
        # Initial board
        self.df = pl.DataFrame(
            board_t_dict,
        )

    def __str__(self) -> str:
        res = io.StringIO()
        with (
            pl.Config(tbl_rows=-1, tbl_cols=-1),
            contextlib.redirect_stdout(res),
        ):
            print(self.df)
        return res.getvalue()

    def step(
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
        colnums = nwise([f"{idx:02}" for idx in range(self.df.width)], 3)

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
                        **dict(zip(colnames, colvalues)),
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
    app.step()
