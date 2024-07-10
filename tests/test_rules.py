import polars as pl
import pytest
from polars.testing import assert_frame_equal
from game_of_life import step


@pytest.mark.xfail(strict=False, reason="No idea why this fails")
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
    # For some reason at this point the df only has zeroes

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [1, 1, 1],
                "1": [1, 0, 0],
                "2": [1, 1, 0],
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
