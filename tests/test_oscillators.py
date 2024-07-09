from collections import OrderedDict
import polars as pl
from polars.testing import assert_frame_equal
from game_of_life import life_step
from run import nwise_wrapping


def test_blinker():
    # Initial board
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0, 0],
            "01": [0, 0, 1, 0, 0],
            "02": [0, 0, 1, 0, 0],
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
                "01": [0, 0, 0, 0, 0],
                "02": [0, 1, 1, 1, 0],
                "03": [0, 0, 0, 0, 0],
                "04": [0, 0, 0, 0, 0],
            }
        ),
    )

    # 2 iterations, should be back to what it was
    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0],
                "01": [0, 0, 1, 0, 0],
                "02": [0, 0, 1, 0, 0],
                "03": [0, 0, 1, 0, 0],
                "04": [0, 0, 0, 0, 0],
            }
        ),
    )


def test_toad():
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0, 0, 0],
            "01": [0, 0, 0, 0, 0, 0],
            "02": [0, 0, 1, 1, 1, 0],
            "03": [0, 1, 1, 1, 0, 0],
            "04": [0, 0, 0, 0, 0, 0],
            "05": [0, 0, 0, 0, 0, 0],
        }
    )

    colnums = nwise_wrapping([f"{idx:02}" for idx in range(df.width)], 3)
    colnames = [cols[1] for cols in colnums]
    colvalues = [life_step(*tuple(cols)) for cols in colnums]

    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0, 0],
                "01": [0, 0, 0, 1, 0, 0],
                "02": [0, 1, 0, 0, 1, 0],
                "03": [0, 1, 0, 0, 1, 0],
                "04": [0, 0, 1, 0, 0, 0],
                "05": [0, 0, 0, 0, 0, 0],
            }
        ),
    )

    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0, 0],
                "01": [0, 0, 0, 0, 0, 0],
                "02": [0, 0, 1, 1, 1, 0],
                "03": [0, 1, 1, 1, 0, 0],
                "04": [0, 0, 0, 0, 0, 0],
                "05": [0, 0, 0, 0, 0, 0],
            }
        ),
    )


def test_toad_wrapping():
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0, 0, 0],
            "01": [0, 0, 0, 0, 0, 0],
            "02": [0, 0, 0, 0, 0, 0],
            "03": [0, 0, 0, 0, 0, 0],
            "04": [1, 1, 0, 0, 0, 1],
            "05": [1, 0, 0, 0, 1, 1],
        }
    )

    colnums = nwise_wrapping([f"{idx:02}" for idx in range(df.width)], 3)
    colnames = [cols[1] for cols in colnums]
    colvalues = [life_step(*tuple(cols)) for cols in colnums]

    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0, 1],
                "01": [0, 0, 0, 0, 0, 0],
                "02": [0, 0, 0, 0, 0, 0],
                "03": [1, 0, 0, 0, 0, 0],
                "04": [0, 1, 0, 0, 1, 0],
                "05": [0, 1, 0, 0, 1, 0],
            }
        ),
    )

    df = df.with_columns(
        **OrderedDict(zip(colnames, colvalues)),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0, 0],
                "01": [0, 0, 0, 0, 0, 0],
                "02": [0, 0, 0, 0, 0, 0],
                "03": [0, 0, 0, 0, 0, 0],
                "04": [1, 1, 0, 0, 0, 1],
                "05": [1, 0, 0, 0, 1, 1],
            }
        ),
    )
