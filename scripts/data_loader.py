from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class DataPreprocessor:
    """Load and preprocess raw POS data for demand forecasting."""

    def __init__(self, input_path: Path, output_path: Path) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.df: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """Run preprocessing steps in sequence and return cleaned DataFrame."""
        self.load_data()
        self.normalize_column_names()
        self.format_dates()
        self.smart_imputation()
        self.drop_unusable_rows()
        self.clean_anomalies()
        self.optimize_dtypes()
        self.export_data()
        return self.df

    def load_data(self) -> None:
        """Read the raw POS CSV from disk."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.input_path}")

        self.df = pd.read_csv(self.input_path)

    def normalize_column_names(self) -> None:
        """Strip leading/trailing spaces from column names."""
        self._ensure_loaded()
        self.df.columns = self.df.columns.str.strip()

    def format_dates(self) -> None:
        """Convert Date column to pandas datetime format robustly."""
        self._ensure_loaded()

        if "Date" not in self.df.columns:
            raise KeyError("Missing required column: 'Date'")

        # Parse mixed date formats while coercing invalid values to NaT.
        self.df["Date"] = pd.to_datetime(
            self.df["Date"],
            errors="coerce",
            dayfirst=False,
        )

    def smart_imputation(self) -> None:
        """Impute missing numeric and categorical values with domain-aware defaults."""
        self._ensure_loaded()

        # Fill target numeric features with 0 when missing.
        numeric_targets = ["Quantity", "SellingPrice"]
        for col in numeric_targets:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        # Fill any other numeric columns with 0 as a safe default.
        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            self.df.loc[:, numeric_columns] = self.df.loc[:, numeric_columns].fillna(0)

        # Fill categorical columns with "Unknown" to preserve rows.
        categorical_columns = self.df.select_dtypes(
            include=["object", "string", "category"]
        ).columns
        if len(categorical_columns) > 0:
            self.df.loc[:, categorical_columns] = self.df.loc[:, categorical_columns].fillna(
                "Unknown"
            )

    def drop_unusable_rows(self) -> None:
        """Drop rows only when all essential product identifiers are missing."""
        self._ensure_loaded()

        essential_identifier_columns = ["Item_ID", "ProductID", "ProductName"]
        present_identifier_columns = [
            col for col in essential_identifier_columns if col in self.df.columns
        ]

        if present_identifier_columns:
            # Normalize empty strings/spaces to NA before dropping.
            self.df.loc[:, present_identifier_columns] = self.df.loc[
                :, present_identifier_columns
            ].replace(r"^\s*$", pd.NA, regex=True)
            self.df = self.df.dropna(subset=present_identifier_columns, how="all")

        # Date cannot be used for time-series learning when missing.
        self.df = self.df.dropna(subset=["Date"])

    def clean_anomalies(self) -> None:
        """Remove rows with invalid sales signals such as non-positive quantity/price."""
        self._ensure_loaded()

        if "Quantity" in self.df.columns:
            self.df = self.df[self.df["Quantity"] > 0]

        if "SellingPrice" in self.df.columns:
            self.df = self.df[self.df["SellingPrice"] > 0]

    def optimize_dtypes(self) -> None:
        """Optimize data types for stable model inputs and memory usage."""
        self._ensure_loaded()

        if "Quantity" in self.df.columns:
            self.df["Quantity"] = pd.to_numeric(self.df["Quantity"], errors="coerce")
            self.df["Quantity"] = self.df["Quantity"].fillna(0).astype("int64")

        if "SellingPrice" in self.df.columns:
            self.df["SellingPrice"] = pd.to_numeric(
                self.df["SellingPrice"], errors="coerce"
            )
            self.df["SellingPrice"] = self.df["SellingPrice"].fillna(0).astype("float64")

    def export_data(self) -> None:
        """Save cleaned output to processed directory."""
        self._ensure_loaded()

        os.makedirs(self.output_path.parent, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "raw" / "DISSANAYAKA POS DATASET.csv"
    output_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"

    preprocessor = DataPreprocessor(input_path=input_file, output_path=output_file)
    cleaned_df = preprocessor.run()

    print("Data preprocessing completed successfully.")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    main()