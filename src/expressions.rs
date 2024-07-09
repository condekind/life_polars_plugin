#![allow(clippy::unused_unit)]
use polars::export::arrow::legacy::utils::CustomIterTools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Int64)]
fn life_step(inputs: &[Series]) -> PolarsResult<Series> {
    let (left, curr, right) = (&inputs[0], &inputs[1], &inputs[2]);
    let (ca_lf, ca_curr, ca_rt) = (left.i64()?, curr.i64()?, right.i64()?);
    let len = ca_curr.len();

    let mut out: Int64Chunked = ca_curr
        .into_no_null_iter()
        .enumerate()
        .map(|(idx, val)| {
            let prev_row = if 0 == idx {
                ca_lf.get(len - 1).unwrap_or(0)
                    + ca_curr.get(len - 1).unwrap_or(0)
                    + ca_rt.get(len - 1).unwrap_or(0)
            } else {
                ca_lf.get(idx - 1).unwrap_or(0)
                    + ca_curr.get(idx - 1).unwrap_or(0)
                    + ca_rt.get(idx - 1).unwrap_or(0)
            };

            // Curr row does not include cell in the middle, a cell is not a neighbour of itself
            let curr_row = ca_lf.get(idx).unwrap_or(0) + ca_rt.get(idx).unwrap_or(0);

            let next_row = if len - 1 == idx {
                ca_lf.get(0).unwrap_or(0) + ca_curr.get(0).unwrap_or(0) + ca_rt.get(0).unwrap_or(0)
            } else {
                ca_lf.get(idx + 1).unwrap_or(0)
                    + ca_curr.get(idx + 1).unwrap_or(0)
                    + ca_rt.get(idx + 1).unwrap_or(0)
            };

            // Life logic
            Some(match (val, prev_row + curr_row + next_row) {
                (1, 2) | (1, 3) => 1,
                (0, 3) => 1,
                _ => 0,
            })
        })
        .collect_trusted();
    out.rename(ca_curr.name());
    Ok(out.into_series())
}
