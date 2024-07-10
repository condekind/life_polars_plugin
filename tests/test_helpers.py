import polars as pl
from polars.testing import assert_frame_equal
from game_of_life import step


def test_step_column_padding():
    """
    This tests whether step(df, n) correctly pads the names of the columns
    """
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0],
            "1": [0, 0, 1, 0, 0],
            "2": [0, 0, 1, 0, 0],
            "3": [0, 0, 1, 0, 0],
            "4": [0, 0, 0, 0, 0],
            "5": [0, 0, 0, 0, 0],
            "6": [0, 0, 0, 0, 0],
            "7": [0, 0, 0, 0, 0],
            "8": [0, 0, 0, 0, 0],
            "9": [0, 0, 0, 0, 0],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0],
                "1": [0, 0, 0, 0, 0],
                "2": [0, 1, 1, 1, 0],
                "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
                "5": [0, 0, 0, 0, 0],
                "6": [0, 0, 0, 0, 0],
                "7": [0, 0, 0, 0, 0],
                "8": [0, 0, 0, 0, 0],
                "9": [0, 0, 0, 0, 0],
            }
        ),
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0],
                "1": [0, 0, 1, 0, 0],
                "2": [0, 0, 1, 0, 0],
                "3": [0, 0, 1, 0, 0],
                "4": [0, 0, 0, 0, 0],
                "5": [0, 0, 0, 0, 0],
                "6": [0, 0, 0, 0, 0],
                "7": [0, 0, 0, 0, 0],
                "8": [0, 0, 0, 0, 0],
                "9": [0, 0, 0, 0, 0],
            }
        ),
    )

    df = step(df, 2)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0],
                "1": [0, 0, 1, 0, 0],
                "2": [0, 0, 1, 0, 0],
                "3": [0, 0, 1, 0, 0],
                "4": [0, 0, 0, 0, 0],
                "5": [0, 0, 0, 0, 0],
                "6": [0, 0, 0, 0, 0],
                "7": [0, 0, 0, 0, 0],
                "8": [0, 0, 0, 0, 0],
                "9": [0, 0, 0, 0, 0],
            }
        ),
    )


def test_step_column_padding_longer():
    """
    This tests whether step(df, n) correctly pads the names of the columns
    """
    df = pl.DataFrame(
        {
            "00": [0, 0, 0, 0, 0],
            "01": [0, 0, 1, 0, 0],
            "02": [0, 0, 1, 0, 0],
            "03": [0, 0, 1, 0, 0],
            "04": [0, 0, 0, 0, 0],
            "05": [0, 0, 0, 0, 0],
            "06": [0, 0, 0, 0, 0],
            "07": [0, 0, 0, 0, 0],
            "08": [0, 0, 0, 0, 0],
            "09": [0, 0, 0, 0, 0],
            "10": [0, 0, 0, 0, 0],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0],
                "01": [0, 0, 0, 0, 0],
                "02": [0, 1, 1, 1, 0],
                "03": [0, 0, 0, 0, 0],
                "04": [0, 0, 0, 0, 0],
                "05": [0, 0, 0, 0, 0],
                "06": [0, 0, 0, 0, 0],
                "07": [0, 0, 0, 0, 0],
                "08": [0, 0, 0, 0, 0],
                "09": [0, 0, 0, 0, 0],
                "10": [0, 0, 0, 0, 0],
            }
        ),
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0],
                "01": [0, 0, 1, 0, 0],
                "02": [0, 0, 1, 0, 0],
                "03": [0, 0, 1, 0, 0],
                "04": [0, 0, 0, 0, 0],
                "05": [0, 0, 0, 0, 0],
                "06": [0, 0, 0, 0, 0],
                "07": [0, 0, 0, 0, 0],
                "08": [0, 0, 0, 0, 0],
                "09": [0, 0, 0, 0, 0],
                "10": [0, 0, 0, 0, 0],
            }
        ),
    )

    df = step(df, 2)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "00": [0, 0, 0, 0, 0],
                "01": [0, 0, 1, 0, 0],
                "02": [0, 0, 1, 0, 0],
                "03": [0, 0, 1, 0, 0],
                "04": [0, 0, 0, 0, 0],
                "05": [0, 0, 0, 0, 0],
                "06": [0, 0, 0, 0, 0],
                "07": [0, 0, 0, 0, 0],
                "08": [0, 0, 0, 0, 0],
                "09": [0, 0, 0, 0, 0],
                "10": [0, 0, 0, 0, 0],
            }
        ),
    )