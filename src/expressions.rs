#![allow(clippy::unused_unit)]
use polars::export::arrow::legacy::utils::CustomIterTools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Int64)]
fn life_step(inputs: &[Series]) -> PolarsResult<Series> {
    let (ca_lf, ca_curr, ca_rt) = (inputs[0].i64()?, inputs[1].i64()?, inputs[2].i64()?);

    let lf = ca_lf
        .cont_slice()
        .expect("Expected input to be contiguous (in a single chunk)");
    let mid = ca_curr
        .cont_slice()
        .expect("Expected input to be contiguous (in a single chunk)");
    let rt = ca_rt
        .cont_slice()
        .expect("Expected input to be contiguous (in a single chunk)");

    let len = lf.len();

    let mut out: Int64Chunked = ca_curr
        .into_no_null_iter()
        .enumerate()
        .map(|(idx, val)| {
            // Neighbours above
            let prev_row = if 0 == idx {
                lf[len - 1] + mid[len - 1] + rt[len - 1]
            } else {
                lf[idx - 1] + mid[idx - 1] + rt[idx - 1]
            };

            // Curr row does not include cell in the middle,
            // a cell is not a neighbour of itself
            let curr_row = lf[idx] + rt[idx];

            // Neighbours below
            let next_row = if len - 1 == idx {
                lf[0] + mid[0] + rt[0]
            } else {
                lf[idx + 1] + mid[idx + 1] + rt[idx + 1]
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
