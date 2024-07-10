import polars as pl
from polars.testing import assert_frame_equal
from game_of_life import step


def test_011110010():
    # Initial board
    df = pl.DataFrame(
        {
            "0": [0, 1, 1],
            "1": [1, 1, 0],
            "2": [0, 1, 0],
        }
    )

    # 1 iteration
    df = step(df)

    # Since the board wraps around both vertically and horizontally, the outcome
    # is not the same as it would be in a larger, empty board (save for the
    # initial pattern), see the larger version of this test.
    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0],
                "1": [0, 0, 0],
                "2": [0, 0, 0],
            }
        ),
    )


def test_011110010_larger():
    # Initial board
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0],
            "1": [0, 0, 1, 1, 0],
            "2": [0, 1, 1, 0, 0],
            "3": [0, 0, 1, 0, 0],
            "4": [0, 0, 0, 0, 0],
        }
    )

    # 1 iteration
    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0],
                "1": [0, 1, 1, 1, 0],
                "2": [0, 1, 0, 0, 0],
                "3": [0, 1, 1, 0, 0],
                "4": [0, 0, 0, 0, 0],
            }
        ),
    )
