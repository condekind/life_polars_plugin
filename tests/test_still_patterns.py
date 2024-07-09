from collections import OrderedDict
import pytest
from game_of_life import life_step
from run import nwise_wrapping

import polars as pl
from polars.testing import assert_frame_equal


def test_block():
    # Initial board
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0],
            "01": [0, 1, 1, 0],
            "02": [0, 1, 1, 0],
            "03": [0, 0, 0, 0],
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
                "00": [0, 0, 0, 0],
                "01": [0, 1, 1, 0],
                "02": [0, 1, 1, 0],
                "03": [0, 0, 0, 0],
            }
        ),
    )


def test_block_wrapping():
    # Initial board
    df = pl.DataFrame(
        {
            "00": [1, 0, 0, 1],
            "01": [0, 0, 0, 0],
            "02": [0, 0, 0, 0],
            "03": [1, 0, 0, 1],
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
                "00": [1, 0, 0, 1],
                "01": [0, 0, 0, 0],
                "02": [0, 0, 0, 0],
                "03": [1, 0, 0, 1],
            }
        ),
    )


@pytest.mark.skip(reason="todo")
def test_beehive():
    pass


@pytest.mark.skip(reason="todo")
def test_loaf():
    pass


@pytest.mark.skip(reason="todo")
def test_boat():
    pass


@pytest.mark.skip(reason="todo")
def test_tub():
    pass
