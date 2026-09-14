from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from pos_pipeline.analysis.abc import analyze_profit_abc
from pos_pipeline.analysis.xyz import analyze_xyz
from pos_pipeline.analysis.abc_xyz import attach_xyz_and_strategy
from logger_config import get_logger


logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="pos_pipeline.analysis.xyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline ABC-XYZ demo using anonymized sample_data/."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("sample_data"),
        help="Directory containing stock.csv and sales.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Directory for demo analysis outputs.",
    )
    parser.add_argument(
        "--abc-months",
        type=int,
        default=12,
        help="Recent months used for ABC value analysis.",
    )
    return parser.parse_args()


def load_sample_data(sample_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_path = sample_dir / "stock.csv"
    sales_path = sample_dir / "sales.parquet"

    if not stock_path.exists():
        raise FileNotFoundError(f"Missing sample stock file: {stock_path}")
    if not sales_path.exists():
        raise FileNotFoundError(f"Missing sample sales file: {sales_path}")

    stock_df = pd.read_csv(stock_path, encoding="utf-8-sig")
    sales_df = pd.read_parquet(sales_path)
    sales_df["rDate"] = pd.to_datetime(sales_df["rDate"].astype(str), format="%Y%m%d")
    return stock_df, sales_df


def build_month_age_map(sales_df: pd.DataFrame) -> dict:
    last_date = sales_df["rDate"].max()
    first_sale_full = (
        sales_df.groupby("GoodsID")["rDate"]
        .min()
        .reset_index()
        .rename(columns={"rDate": "FirstSaleDate"})
    )
    month_age = (
        (last_date.year - first_sale_full["FirstSaleDate"].dt.year) * 12
        + (last_date.month - first_sale_full["FirstSaleDate"].dt.month)
    ).clip(lower=1)
    return dict(zip(first_sale_full["GoodsID"], month_age))


def run_demo(
    stock_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    abc_months: int,
) -> pd.DataFrame:
    last_date = sales_df["rDate"].max()
    start_date_abc = last_date - pd.DateOffset(months=abc_months)
    sales_recent = sales_df[sales_df["rDate"] >= start_date_abc].copy()
    month_age_map = build_month_age_map(sales_df)

    abc_df = analyze_profit_abc(
        stock_df,
        sales_recent,
        month_age_map=month_age_map,
    )
    return attach_xyz_and_strategy(abc_df, analyze_xyz(sales_df))


def print_summary(result_df: pd.DataFrame) -> None:
    print("\n=== Offline Demo Summary ===")
    print(f"Rows: {len(result_df)}")
    print("\nABC class counts:")
    print(result_df["ABC_Class"].value_counts(dropna=False).to_string())
    print("\nXYZ class counts:")
    print(result_df["XYZ_Class"].value_counts(dropna=False).to_string())


def main() -> None:
    args = parse_args()

    logger.info("Loading sample data from %s ...", args.sample_dir)
    stock_df, sales_df = load_sample_data(args.sample_dir)

    logger.info("Running ABC-XYZ demo analysis ...")
    result_df = run_demo(stock_df, sales_df, abc_months=args.abc_months)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "abc_xyz_analysis.csv"
    try:
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / f"abc_xyz_analysis_{timestamp}.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print("Default output file was locked; wrote a timestamped file instead.")

    logger.info("Demo analysis complete. Rows: %d", len(result_df))
    logger.info("Output written to: %s", output_path)
    print(f"\nOutput written to: {output_path}")
    print_summary(result_df)


if __name__ == "__main__":
    main()
