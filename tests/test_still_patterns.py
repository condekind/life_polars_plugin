from game_of_life import step

import polars as pl
from polars.testing import assert_frame_equal


def test_block():
    # Initial board
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0],
            "1": [0, 1, 1, 0],
            "2": [0, 1, 1, 0],
            "3": [0, 0, 0, 0],
        }
    )

    # 1 iteration
    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0],
                "1": [0, 1, 1, 0],
                "2": [0, 1, 1, 0],
                "3": [0, 0, 0, 0],
            }
        ),
    )


def test_block_wrapping():
    df = pl.DataFrame(
        {
            "0": [1, 0, 0, 1],
            "1": [0, 0, 0, 0],
            "2": [0, 0, 0, 0],
            "3": [1, 0, 0, 1],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [1, 0, 0, 1],
                "1": [0, 0, 0, 0],
                "2": [0, 0, 0, 0],
                "3": [1, 0, 0, 1],
            }
        ),
    )


def test_beehive():
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0, 0],
            "1": [0, 0, 1, 1, 0, 0],
            "2": [0, 1, 0, 0, 1, 0],
            "3": [0, 0, 1, 1, 0, 0],
            "4": [0, 0, 0, 0, 0, 0],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [0, 0, 1, 1, 0, 0],
                "2": [0, 1, 0, 0, 1, 0],
                "3": [0, 0, 1, 1, 0, 0],
                "4": [0, 0, 0, 0, 0, 0],
            }
        ),
    )
