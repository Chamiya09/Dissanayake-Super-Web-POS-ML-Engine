from __future__ import annotations

import argparse
from pathlib import Path

from member1_data_cleaning import (
    apply_price_cleaning,
    load_and_validate,
    resolve_input_with_fallback,
    resolve_path,
)
from member2_eda import run_eda_plots
from member3_feature_engineering import build_monthly_features, build_weekly_features, load_cleaned_data
from member4_data_preparation import build_feature_columns, load_feature_data
from member5_modeling import train_two_stage
from member6_evaluation import build_accuracy_table, plot_top_product, save_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Team pipeline runner: Data Cleaning -> EDA -> Feature Engineering -> Data Preparation -> Modeling -> Evaluation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/DISSANAYAKA_POS_DATASET_2018-2025.csv"),
        help="Raw input dataset path",
    )
    parser.add_argument(
        "--output-cleaned",
        type=Path,
        default=Path("data/processed/cleaned_pos_data.csv"),
        help="Output path for cleaned dataset",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("exports/plots"),
        help="Directory to save EDA plots",
    )
    parser.add_argument(
        "--weekly-output",
        type=Path,
        default=Path("data/processed/final_weekly_features.csv"),
        help="Output path for weekly feature dataset",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=Path("data/processed/final_monthly_features.csv"),
        help="Output path for monthly feature dataset",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/dissanayaka_master_model.pkl"),
        help="Path to save trained full pipeline",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("exports/plots/final_forecast_vs_actual_top_product_2025.png"),
        help="Path to save forecast-vs-actual chart",
    )
    return parser.parse_args()


def run_pipeline(
    input_path: Path,
    output_cleaned: Path,
    plots_dir: Path,
    weekly_output: Path,
    monthly_output: Path,
    model_output: Path,
    plot_output: Path,
) -> None:
    cleaned = load_and_validate(input_path)
    cleaned = apply_price_cleaning(cleaned)

    output_cleaned.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_cleaned, index=False)
    print(f"Saved cleaned dataset: {output_cleaned}")

    run_eda_plots(cleaned, plots_dir)
    print(f"Saved plots directory: {plots_dir}")

    cleaned_for_features = load_cleaned_data(output_cleaned)
    weekly = build_weekly_features(cleaned_for_features)
    monthly = build_monthly_features(cleaned_for_features)

    weekly_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(weekly_output, index=False)
    monthly.to_csv(monthly_output, index=False)

    print(f"Saved weekly features: {weekly_output} | rows={len(weekly):,}")
    print(f"Saved monthly features: {monthly_output} | rows={len(monthly):,}")

    weekly_df = load_feature_data(weekly_output, date_col="Week_Ending_Sunday")
    monthly_df = load_feature_data(monthly_output, date_col="Month_Start")

    weekly_features = build_feature_columns(weekly_df)
    monthly_features = build_feature_columns(monthly_df)
    if not weekly_features:
        raise ValueError("No numeric weekly feature columns found.")
    if not monthly_features:
        raise ValueError("No numeric monthly feature columns found.")

    weekly_result = train_two_stage(weekly_df, date_col="Week_Ending_Sunday", feature_cols=weekly_features)
    monthly_result = train_two_stage(monthly_df, date_col="Month_Start", feature_cols=monthly_features)

    table = build_accuracy_table(weekly_result["metrics"], monthly_result["metrics"])
    print("\nFull Scale Accuracy Table")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_top_product(weekly_result["predictions"], monthly_result["predictions"], plot_output)
    print(f"\nSaved forecast plot: {plot_output}")

    save_model_artifact(weekly_result, monthly_result, model_output)
    print(f"Saved model: {model_output}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    input_path = resolve_input_with_fallback(resolve_path(args.input, project_root))
    output_cleaned = resolve_path(args.output_cleaned, project_root)
    plots_dir = resolve_path(args.plots_dir, project_root)
    weekly_output = resolve_path(args.weekly_output, project_root)
    monthly_output = resolve_path(args.monthly_output, project_root)
    model_output = resolve_path(args.model_output, project_root)
    plot_output = resolve_path(args.plot_output, project_root)

    run_pipeline(
        input_path=input_path,
        output_cleaned=output_cleaned,
        plots_dir=plots_dir,
        weekly_output=weekly_output,
        monthly_output=monthly_output,
        model_output=model_output,
        plot_output=plot_output,
    )


if __name__ == "__main__":
    main()
