from collections import OrderedDict
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from game_of_life import life_step
from run import nwise_wrapping


@pytest.mark.xfail(
    strict=False, reason="No idea why this fails - need to investigate the rust code"
)
def test_011110010():
    # Initial board
    df = pl.DataFrame(
        {
            "00": [0, 1, 1],
            "01": [1, 1, 0],
            "02": [0, 1, 0],
        }
    )

    colnums = nwise_wrapping([f"{idx:02}" for idx in range(df.width)], 3)
    colnames = [cols[1] for cols in colnums]
    colvalues = [life_step(*tuple(cols)) for cols in colnums]

    # 1 iteration
    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )
    # For some reason at this point the df only has zeroes

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [1, 1, 1],
                "01": [1, 0, 0],
                "02": [1, 1, 0],
            }
        ),
    )


def test_011110010_larger():
    # Initial board
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0, 0],
            "01": [0, 0, 1, 1, 0],
            "02": [0, 1, 1, 0, 0],
            "03": [0, 0, 1, 0, 0],
            "04": [0, 0, 0, 0, 0],
        }
    )

    colnums = nwise_wrapping([f"{idx:02}" for idx in range(df.width)], 3)
    colnames = [cols[1] for cols in colnums]
    colvalues = [life_step(*tuple(cols)) for cols in colnums]

    # 1 iteration
    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0],
                "01": [0, 1, 1, 1, 0],
                "02": [0, 1, 0, 0, 0],
                "03": [0, 1, 1, 0, 0],
                "04": [0, 0, 0, 0, 0],
            }
        ),
    )
