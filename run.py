import argparse
import contextlib
import io
import sys
from time import sleep

from game_of_life import parse_board, board_to_df, step
import polars as pl


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

        if print_df:
            print(self)

        iteration_cnt = 0
        try:
            for _ in range(n):
                self.df = step(self.df)
                iteration_cnt += 1
                if print_df:
                    # Clear screen
                    print("\033[2J")
                    print(self)
                sleep(delay)

        except KeyboardInterrupt:
            print(
                f"\nKeyboard Interrupt: ran for {iteration_cnt} iterations. Aborting..."
            )
            print(f"max_num_steps={self._args.num_steps}\ndelay={self._args.delay}")


if __name__ == "__main__":
    app = Application()
    app.start()
