# Dissanayaka Super Web POS - ML Engine

FastAPI demand forecasting service for the Dissanayaka Super Web POS system. The ML engine trains demand models from POS sales data and exposes forecast endpoints used by inventory planning and dashboard features.

## Tech Stack

- Python 3.11 recommended
- FastAPI
- Uvicorn
- Pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- Joblib
- Matplotlib and Seaborn
- PostgreSQL / SQLAlchemy support

## Main Features

- Demand forecast API for weekly and monthly product demand.
- Model health endpoint for operational checks.
- Training pipeline for cleaning raw POS data, creating features, and generating model artifacts.
- Processed weekly and monthly feature datasets.
- Configurable model path and log-target prediction behavior.

## Prerequisites

- Python 3.11.x recommended
- pip
- A raw POS dataset CSV if you need to train locally
- Generated model and processed feature files if you only need to run the API

## Quick Start

### Windows PowerShell

```powershell
cd "D:\Project\GitHub Project\Dissanayaka Super Web POS\Dissanayake-Super-Web-POS-ML-Engine"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
cd Dissanayake-Super-Web-POS-ML-Engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API starts on:

```text
http://127.0.0.1:8000
```

## Environment Variables

Copy `.env.example` to `.env` and adjust values as needed:

```env
APP_ENV=development
DATABASE_URL=your_database_url_here
MODEL_PATH=models/dissanayaka_master_model.pkl
MODEL_USES_LOG_TARGET=true
```

Variable meanings:

- `APP_ENV` - runtime environment label, usually `development`.
- `DATABASE_URL` - optional database connection string for database-backed workflows.
- `MODEL_PATH` - path to the trained model artifact.
- `MODEL_USES_LOG_TARGET` - set to `true` when the trained model predicts log-transformed demand.

## Required Generated Files

The repository ignores datasets, processed data, and trained model artifacts. To run forecasts, create or copy these files:

```text
models/dissanayaka_master_model.pkl
data/processed/final_weekly_features.csv
data/processed/final_monthly_features.csv
```

If these files are missing, run the training pipeline.

## Training The Model

Default expected raw dataset path:

```text
data/raw/DISSANAYAKA_POS_DATASET_2018-2025.csv
```

Run training with an explicit input path:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe scripts\team_pipeline\main.py --input "D:\Project\GitHub Project\Dissanayaka Super Web POS\DISSANAYAKA_POS_DATASET_2018-2025.csv"
```

### macOS / Linux

```bash
python scripts/team_pipeline/main.py --input "/path/to/DISSANAYAKA_POS_DATASET_2018-2025.csv"
```

After training completes, the pipeline should generate the model and processed feature files used by the API.

## API Endpoints

```text
GET /health
GET /api/model-health
GET /api/forecast?product_id=PI00001&timeframe=weekly
GET /api/forecast?product_id=PI00001&timeframe=monthly
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

## Project Structure

```text
app/
  main.py              Primary FastAPI app used by Uvicorn
  api/                 Additional API module and route code

scripts/
  team_pipeline/       Training and feature engineering pipeline

data/
  raw/                 Local raw datasets, ignored by Git
  processed/           Generated feature files, ignored by Git

models/                Generated model artifacts, ignored by Git
exports/               Generated plots and outputs, ignored by Git
```

## Development Workflow

1. Create and activate `.venv`.
2. Install dependencies.
3. Copy `.env.example` to `.env`.
4. Train the model or copy existing generated files.
5. Start Uvicorn.
6. Test `/health`, `/api/model-health`, and `/api/forecast`.

## Troubleshooting

### `ModuleNotFoundError`

The virtual environment is not active or dependencies are not installed:

```powershell
pip install -r requirements.txt
```

### `Model artifact not found`

Train the model or place the artifact at:

```text
models/dissanayaka_master_model.pkl
```

### Forecast returns product not found

Use the product ID format found in the processed feature files, for example:

```text
PI00001
```

### CSV parser errors during training

Pass the dataset path explicitly with `--input` and confirm the CSV headers match the training pipeline expectations.

## Deployment Notes

- Do not commit `.env`, raw datasets, processed datasets, or trained model artifacts.
- Provide `MODEL_PATH` and generated files through deployment storage.
- Run the API with a production ASGI server configuration instead of `--reload`.
