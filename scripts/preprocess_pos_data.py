from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class DataPreprocessor:
    """Load, clean, and export retail POS data."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        essential_columns: Iterable[str] = ("Item_ID", "Transaction_ID"),
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.essential_columns = list(essential_columns)
        self.df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Read raw CSV data from disk."""
        self.df = pd.read_csv(self.input_path)

    def format_dates(self) -> None:
        """Convert the Date column to pandas datetime."""
        self._ensure_dataframe_loaded()

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

    def handle_missing_values(self) -> None:
        """Fill numeric nulls with 0 and drop rows missing essential IDs."""
        self._ensure_dataframe_loaded()

        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        self.df.loc[:, numeric_columns] = self.df.loc[:, numeric_columns].fillna(0)

        present_essential_columns = [
            col for col in self.essential_columns if col in self.df.columns
        ]
        if present_essential_columns:
            self.df = self.df.dropna(subset=present_essential_columns)

    def clean_anomalies(self) -> None:
        """Remove invalid sales rows with non-positive quantity or selling price."""
        self._ensure_dataframe_loaded()

        if "Quantity" in self.df.columns:
            self.df = self.df[self.df["Quantity"] > 0]

        if "SellingPrice" in self.df.columns:
            self.df = self.df[self.df["SellingPrice"] > 0]

    def export_data(self) -> None:
        """Write cleaned data to the processed directory."""
        self._ensure_dataframe_loaded()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)

    def run_pipeline(self) -> pd.DataFrame:
        """Execute all preprocessing steps in the required order."""
        self.load_data()
        self.format_dates()
        self.handle_missing_values()
        self.clean_anomalies()
        self.export_data()
        return self.df

    def _ensure_dataframe_loaded(self) -> None:
        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "raw" / "raw_pos_data.csv"
    output_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"

    preprocessor = DataPreprocessor(input_path=input_file, output_path=output_file)
    cleaned_df = preprocessor.run_pipeline()

    print(f"Preprocessing complete. Rows: {len(cleaned_df)}")
    print(f"Cleaned data saved to: {output_file}")


if __name__ == "__main__":
    main()