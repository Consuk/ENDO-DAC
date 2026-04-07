import os
import argparse
import numpy as np
import pandas as pd


DEFAULT_METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def _to_float_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _weighted_aggregate(df, metric_cols, weight_col=None):
    use_weighted = weight_col is not None and weight_col in df.columns

    if use_weighted:
        w = _to_float_series(df, weight_col).fillna(0.0).clip(lower=0.0)
        weight_sum = float(w.sum())
        if weight_sum > 0:
            out = {weight_col: weight_sum}
            for c in metric_cols:
                v = _to_float_series(df, c)
                mask = v.notna() & w.notna()
                if not mask.any():
                    out[c] = np.nan
                    continue
                ww = w[mask]
                ww_sum = float(ww.sum())
                if ww_sum <= 0:
                    out[c] = float(v[mask].mean())
                else:
                    out[c] = float((v[mask] * ww).sum() / ww_sum)
            return out

    out = {}
    if weight_col is not None:
        out[weight_col] = (
            float(_to_float_series(df, weight_col).fillna(0.0).sum())
            if weight_col in df.columns
            else float(len(df))
        )
    for c in metric_cols:
        out[c] = float(_to_float_series(df, c).mean())
    return out


def main():
    parser = argparse.ArgumentParser("Average corruption metrics CSV")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV (typically summary_by_severity.csv from corruption evaluation).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where output CSV files are written.",
    )
    parser.add_argument(
        "--global_filename",
        type=str,
        default="global_average.csv",
        help="Filename for global averages CSV.",
    )
    parser.add_argument(
        "--per_corruption_filename",
        type=str,
        default="corruption_averages.csv",
        help="Filename for per-corruption averages CSV.",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default="corruption",
        help="Column used for per-corruption grouping.",
    )
    parser.add_argument(
        "--weight_col",
        type=str,
        default="num_samples",
        help="Weight column for weighted averaging. If missing, falls back to unweighted means.",
    )
    parser.add_argument(
        "--metric_cols",
        nargs="+",
        default=None,
        help="Optional metric columns override. Default: abs_rel sq_rel rmse rmse_log a1 a2 a3",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    metric_cols = args.metric_cols if args.metric_cols else list(DEFAULT_METRICS)
    if "avg_inference_ms" in df.columns and "avg_inference_ms" not in metric_cols:
        metric_cols = ["avg_inference_ms"] + metric_cols

    missing_metrics = [c for c in metric_cols if c not in df.columns]
    if missing_metrics:
        raise ValueError(
            f"Missing metric columns in input CSV: {missing_metrics}. "
            f"Available columns: {list(df.columns)}"
        )

    global_stats = _weighted_aggregate(df, metric_cols, args.weight_col)
    global_columns = ["label"]
    if args.weight_col is not None:
        global_columns.append(args.weight_col)
    global_columns.extend(metric_cols)
    global_row = ["global"]
    if args.weight_col is not None:
        global_row.append(global_stats.get(args.weight_col, np.nan))
    global_row.extend([global_stats[m] for m in metric_cols])
    global_df = pd.DataFrame([global_row], columns=global_columns)

    global_out = os.path.join(args.output_dir, args.global_filename)
    global_df.to_csv(global_out, index=False)

    print("=== Global Average ===")
    if args.weight_col in global_stats:
        print(f"{args.weight_col:16s}: {global_stats[args.weight_col]:.1f}")
    for metric in metric_cols:
        print(f"{metric:16s}: {global_stats[metric]:.6f}")
    print(f"Saved: {global_out}")

    if args.group_col in df.columns:
        rows = []
        for group_name, gdf in df.groupby(args.group_col, sort=True):
            gstats = _weighted_aggregate(gdf, metric_cols, args.weight_col)
            row = [group_name]
            if args.weight_col is not None:
                row.append(gstats.get(args.weight_col, np.nan))
            row.extend([gstats[m] for m in metric_cols])
            rows.append(row)

        per_cols = [args.group_col]
        if args.weight_col is not None:
            per_cols.append(args.weight_col)
        per_cols.extend(metric_cols)
        per_df = pd.DataFrame(rows, columns=per_cols)

        corr_out = os.path.join(args.output_dir, args.per_corruption_filename)
        per_df.to_csv(corr_out, index=False)
        print(f"Saved: {corr_out}")


if __name__ == "__main__":
    main()
