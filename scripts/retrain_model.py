from __future__ import annotations


"""
Background retraining entry point.

Suggested production flow:
1. Pull recent data from Neon PostgreSQL.
2. Run preprocessing + feature generation.
3. Train and validate model.
4. Persist model artifact to models/.
5. Optionally register model metadata/version in DB.
"""


def main() -> None:
    print("Retraining job started...")
    # TODO: implement data extraction from Neon PostgreSQL
    # TODO: implement model training and artifact save
    print("Retraining job finished.")


if __name__ == "__main__":
    main()
