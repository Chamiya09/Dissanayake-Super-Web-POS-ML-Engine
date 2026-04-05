from __future__ import annotations

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
        self.format_dates()
        self.handle_missing_values()
        self.clean_anomalies()
        self.export_data()
        return self.df

    def load_data(self) -> None:
        """Read the raw POS CSV from disk."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.input_path}")

        self.df = pd.read_csv(self.input_path)

    def format_dates(self) -> None:
        """Convert Date column to pandas datetime format."""
        self._ensure_loaded()

        if "Date" not in self.df.columns:
            raise KeyError("Missing required column: 'Date'")

        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

    def handle_missing_values(self) -> None:
        """Fill numeric nulls and drop rows missing essential IDs."""
        self._ensure_loaded()

        # Fill all numeric columns (e.g., Price, SellingPrice, Quantity) with zero.
        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        self.df.loc[:, numeric_columns] = self.df.loc[:, numeric_columns].fillna(0)

        # Drop rows with missing identifiers used to track products and transactions.
        essential_columns = ["Item_ID", "Transaction_ID"]
        present_essential_columns = [
            col for col in essential_columns if col in self.df.columns
        ]

        if present_essential_columns:
            self.df = self.df.dropna(subset=present_essential_columns)

    def clean_anomalies(self) -> None:
        """Remove rows with invalid sales signals such as non-positive quantity/price."""
        self._ensure_loaded()

        if "Quantity" in self.df.columns:
            self.df = self.df[self.df["Quantity"] > 0]

        if "SellingPrice" in self.df.columns:
            self.df = self.df[self.df["SellingPrice"] > 0]

    def export_data(self) -> None:
        """Save cleaned output to processed directory."""
        self._ensure_loaded()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)

    def _ensure_loaded(self) -> None:
        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "raw" / "raw_pos_data.csv"
    output_file = project_root / "data" / "processed" / "cleaned_pos_data.csv"

    preprocessor = DataPreprocessor(input_path=input_file, output_path=output_file)
    cleaned_df = preprocessor.run()

    print("Data preprocessing completed successfully.")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    main()