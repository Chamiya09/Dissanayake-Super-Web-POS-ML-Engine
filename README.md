# Dissanayake-Super-Web-POS-ML-Engine

FastAPI-based demand forecasting service for the Dissanayake Super Web POS system.  
This project contains:

- the ML API used by the frontend dashboard
- the training pipeline that prepares features and trains the forecast model

## Prerequisites

- Python `3.11` recommended
- `git`
- A local copy of the raw dataset CSV if you want to train the model

Recommended Python version:

```text
Python 3.11.x
```

## Clone The Repository

```bash
git clone <your-github-repository-url>
cd Dissanayake-Super-Web-POS-ML-Engine
```

## Create A Virtual Environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Mac / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Setup

Copy the example environment file:

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Mac / Linux

```bash
cp .env.example .env
```

Current environment variables:

- `APP_ENV` - application environment, usually `development`
- `DATABASE_URL` - optional database connection string
- `MODEL_PATH` - path to the trained model artifact
- `MODEL_USES_LOG_TARGET` - set to `true` if the trained model predicts log-transformed demand, otherwise `false`

## Important Note About Shared Files

This repository ignores:

- virtual environments
- local `.env` files
- raw / processed datasets
- trained model artifacts

That means a teammate can clone the repo and install dependencies immediately, but they must also do one of these before the ML API can return forecasts:

1. place the trained model and processed feature files in the expected folders, or
2. run the training pipeline locally to generate them

Expected generated files:

- `models/dissanayaka_master_model.pkl`
- `data/processed/final_weekly_features.csv`
- `data/processed/final_monthly_features.csv`

## Training The Model Locally

The training pipeline expects a raw CSV dataset. By default it looks for:

```text
data/raw/DISSANAYAKA_POS_DATASET_2018-2025.csv
```

If your file is somewhere else, pass it explicitly:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe scripts\team_pipeline\main.py --input "C:\path\to\DISSANAYAKA_POS_DATASET_2018-2025.csv"
```

### Mac / Linux

```bash
python scripts/team_pipeline/main.py --input "/path/to/DISSANAYAKA_POS_DATASET_2018-2025.csv"
```

When the pipeline finishes successfully, it generates the processed feature files and trained model required by the API.

## Start The Development Server

After the model artifact and processed feature files exist, start the FastAPI server:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Mac / Linux

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API will be available at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /health`
- `GET /api/forecast?product_id=PI00001&timeframe=weekly`
- `GET /api/forecast?product_id=PI00001&timeframe=monthly`
- `GET /api/model-health`

## Typical First-Time Setup Flow

1. Clone the repo.
2. Create and activate `.venv`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env`.
5. Run the training pipeline if model/data artifacts are not already available.
6. Start the FastAPI server with Uvicorn.

## Troubleshooting

### `ModuleNotFoundError`

Dependencies are not installed in the active virtual environment.

```bash
pip install -r requirements.txt
```

### `Model artifact not found`

Run the training pipeline first, or place the trained model file in:

```text
models/dissanayaka_master_model.pkl
```

### `Product ID not found`

The API expects product IDs in the trained dataset format, for example:

```text
PI00001
```

### CSV parser errors during training

Use the updated pipeline loader and pass the dataset path explicitly with `--input`.
