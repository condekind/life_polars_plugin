import polars as pl
from polars.testing import assert_frame_equal
from game_of_life import step


def test_blinker():
    # Initial board
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0],
            "1": [0, 0, 1, 0, 0],
            "2": [0, 0, 1, 0, 0],
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
                "1": [0, 0, 0, 0, 0],
                "2": [0, 1, 1, 1, 0],
                "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
            }
        ),
    )

    # 2 iterations, should be back to what it was
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
            }
        ),
    )

    # Step 2 at once, should stay the same
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
            }
        ),
    )


def test_toad():
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0, 0],
            "1": [0, 0, 0, 0, 0, 0],
            "2": [0, 0, 1, 1, 1, 0],
            "3": [0, 1, 1, 1, 0, 0],
            "4": [0, 0, 0, 0, 0, 0],
            "5": [0, 0, 0, 0, 0, 0],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [0, 0, 0, 1, 0, 0],
                "2": [0, 1, 0, 0, 1, 0],
                "3": [0, 1, 0, 0, 1, 0],
                "4": [0, 0, 1, 0, 0, 0],
                "5": [0, 0, 0, 0, 0, 0],
            }
        ),
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [0, 0, 0, 0, 0, 0],
                "2": [0, 0, 1, 1, 1, 0],
                "3": [0, 1, 1, 1, 0, 0],
                "4": [0, 0, 0, 0, 0, 0],
                "5": [0, 0, 0, 0, 0, 0],
            }
        ),
    )


def test_toad_wrapping():
    df = pl.DataFrame(
        {
            "0": [0, 0, 0, 0, 0, 0],
            "1": [0, 0, 0, 0, 0, 0],
            "2": [0, 0, 0, 0, 0, 0],
            "3": [0, 0, 0, 0, 0, 0],
            "4": [1, 1, 0, 0, 0, 1],
            "5": [1, 0, 0, 0, 1, 1],
        }
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0, 1],
                "1": [0, 0, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0, 0],
                "3": [1, 0, 0, 0, 0, 0],
                "4": [0, 1, 0, 0, 1, 0],
                "5": [0, 1, 0, 0, 1, 0],
            }
        ),
    )

    df = step(df)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [0, 0, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0, 0],
                "3": [0, 0, 0, 0, 0, 0],
                "4": [1, 1, 0, 0, 0, 1],
                "5": [1, 0, 0, 0, 1, 1],
            }
        ),
    )
