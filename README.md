# Sales Demand Forecasting with Feature Store

## Overview

A major retail chain transforms sales demand forecasting across multiple stores/products using LightGBM + Time Series Features, improving cash flow through better inventory management while increasing product availability and satisfaction.

This example demonstrates **Feature Store for production ML feature serving** using real Walmart sales data. 

### Business Context

**Use Case**: Major retail chain transforms sales demand forecasting using LightGBM + Time Series Features

**Dataset**: Real Walmart Sales Forecasting Data from Kaggle
- 421,000 weekly sales records
- 45 stores × 99 departments
- 143 weeks of data (2010-2012)
- External factors: holidays, markdowns, temperature, economic indicators

**Business Impact**:
- Store Managers: Better inventory planning, reduced stockouts
- Procurement: Optimized ordering, improved supplier negotiations
- Finance: Improved cash flow through better inventory management
- Customers: Higher product availability and satisfaction

---

## Why Feature Store?

### Key Problems Solved

1. **Training-Serving Skew**
   - Problem: Features computed differently in training vs production
   - Feature Store Solution: Same feature definitions and code path for offline (training) and online (serving)

2. **Point-in-Time Correctness**
   - Problem: Data leakage from accidentally using future information
   - Feature Store Solution: Automatic point-in-time joins prevent data leakage

3. **Feature Discovery & Reusability**
   - Problem: Teams duplicate effort, features hard to discover
   - Feature Store Solution: Central registry with rich metadata, tags, owners

4. **Fast Online Serving**
   - Problem: Production models need features in milliseconds
   - Feature Store Solution: Materialized online store for low-latency retrieval (<10ms)

5. **Feature Engineering Pattern**
   - Problem: Complex time-series features need careful computation
   - Feature Store Solution: Pre-compute in ETL pipeline, serve consistently via Feature Store

---

## Architecture: Standard ETL + Feature Store Pattern

```
┌─────────────────────────┐
│  Raw CSV Data (Kaggle)  │
│  • train.csv            │
│  • features.csv         │
│  • stores.csv           │
└────────┬────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│  ETL Pipeline (download_data.py)                          │
│  • Compute time-series features                           │
│    - Lags (sales_lag_1, sales_lag_2, sales_lag_4)         │
│    - Rolling avgs (sales_rolling_mean_4, _mean_12)        │
│    - Rolling std (sales_rolling_std_4)                    │
│  • Merge external factors                                 │
│  • Save to Parquet                                        │
└────────┬──────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│  Feature Store Offline Store (Parquet Files)                      │
│  • sales_features.parquet (sales + time-series features)  │
│  • store_features.parquet (external factors)              │
└────────┬──────────────────────────────────────────────────┘
         │
         ▼         Training                  Inference
   ┌─────┴─────┐
   │           ▼
   │  ┌──────────────────────┐      ┌──────────────────────┐
   │  │ get_historical_      │      │ feast materialize    │
   │  │ features()           │      │ (offline → online)   │
   │  │ + On-Demand          │      └────────┬─────────────┘
   │  │   Transformations    │               │
   │  └──────────┬───────────┘               ▼
   │             │               ┌──────────────────────────┐
   │             ▼               │  Online Store (SQLite)   │
   │  ┌──────────────────────┐  │  • Pre-materialized      │
   │  │  Model Training      │  │  • Fast lookup (<10ms)   │
   │  └──────────────────────┘  └────────┬─────────────────┘
   │                                      │
   └──────────────────────────────────────▼
                               ┌──────────────────────────┐
                               │ get_online_features()    │
                               │ + On-Demand              │
                               │   Transformations        │
                               └────────┬─────────────────┘
                                        │
                                        ▼
                               ┌──────────────────────────┐
                               │  Model Inference         │
                               └──────────────────────────┘
```

**Key Points:**
1. **ETL Pre-Computation**: Time-series features computed once in `download_data.py`
2. **Feature Store Serving**: Ensures training-serving consistency via same feature definitions
3. **On-Demand Transformations**: Applied at retrieval time (both training and serving)
4. **Materialization**: Push features to online store for low-latency inference

---

## Features in This Example

### Entities
- **`store`**: Store ID (1-45) - Physical retail location
- **`dept`**: Department ID (1-99) - Product category within store
- Feature Store automatically handles composite keys with multiple entities

### Base Features (15 features)
From Kaggle datasets:
- `weekly_sales` - Raw sales amount (target variable)
- `is_holiday` - Binary indicator for holiday weeks
- `temperature`, `fuel_price`, `cpi`, `unemployment` - Economic indicators
- `markdown1` through `markdown5` - Promotional markdowns
- `total_markdown`, `has_markdown` - Markdown aggregations
- `store_type`, `store_size` - Store characteristics

### Pre-Computed Time-Series Features (6 features)
**Computed in ETL pipeline** (`download_data.py`):
- `sales_lag_1`, `sales_lag_2`, `sales_lag_4` - Historical sales lags
- `sales_rolling_mean_4`, `sales_rolling_mean_12` - Rolling averages
- `sales_rolling_std_4` - Rolling standard deviation (volatility)

### On-Demand Transformations (7 features)
**Computed at retrieval time** (both training and serving):
- Normalization: `sales_normalized`, `temperature_normalized`
- Interactions: `sales_per_sqft`, `markdown_efficiency`
- Temporal: `sales_velocity`, `sales_acceleration`, `demand_stability_score`

**Total: 28 features** (15 base + 6 time-series + 7 on-demand)

---

## Feature Engineering Pattern

### Standard ETL Pattern (Production-Ready)

This example uses the standard production pattern for feature engineering:

**Step 1: Pre-Compute Features in ETL Pipeline**
```python
# download_data.py
for store_dept, group in df.groupby(['Store', 'Dept']):
    group['sales_lag_1'] = group['Weekly_Sales'].shift(1)
    group['sales_rolling_mean_4'] = group['Weekly_Sales'].rolling(4).mean()
    # ... more features
    
df.to_parquet('sales_features.parquet')  # Save to offline store
```

**Step 2: Feature Store Serves Features Consistently**
```python
# Training
training_features = store.get_historical_features(
    entity_df=training_entities,
    features=["sales_history_features:*", "store_external_features:*"]
)

# Inference (after materialization)
online_features = store.get_online_features(
    entity_rows=[{"store": 1, "dept": 5}],
    features=["sales_history_features:*", "store_external_features:*"]
)
```
---

## Project Structure

```
sales-demand-forecasting/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies  
├── download_data.py                   # Download & compute features from Kaggle
├── demo.py                            # Demonstration script
├── app.py                             # Interactive Streamlit application
└── feature_repo/                      # Feature Store feature repository
    ├── __init__.py
    ├── feature_store.yaml             # Feature Store configuration (local provider)
    ├── features.py                    # Feature definitions + on-demand transformations
    ├── feast_registry.db              # (generated by feast apply)
    └── data/                          # Data directory
        ├── train.csv                  # (downloaded from Kaggle)
        ├── features.csv               # (downloaded from Kaggle)
        ├── stores.csv                 # (downloaded from Kaggle)
        ├── sales_features.parquet     # (pre-computed features)
        └── store_features.parquet     # (external factors + store metadata)
```

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Kaggle Account**: For downloading the Walmart dataset

### Installation

```bash
# Install dependencies (includes Feast, Pandas, LightGBM, Streamlit)
pip install -r requirements.txt
```

### Kaggle Setup

1. Create a Kaggle account at https://www.kaggle.com
2. Generate API credentials:
   - Go to Account Settings → API → Create New API Token
   - This downloads `kaggle.json`
3. Place credentials:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Usage

### Step 1: Download Dataset and Compute Features

```bash
python download_data.py
```

This script:
- Downloads Walmart dataset from Kaggle (train.csv, features.csv, stores.csv)
- Computes time-series features (lags, rolling averages, rolling std)
- Merges external factors and store metadata
- Saves to Parquet files (421K records with features)

### Step 2: Apply Feature Store Features

```bash
cd feature_repo
feast apply
cd ..
```

This registers:
- Entities: `store`, `dept` (composite key)
- Feature Views: `sales_history_features`, `store_external_features`
- On-Demand Feature Views: `feature_transformations`, `temporal_transformations`
- Feature Service: `demand_forecasting_service`

### Step 3: Materialize Features to Online Store

Materialize features from offline store to online store for fast inference:

```bash
cd feature_repo

# Materialize all features up to current time
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

cd ..
```

This command:
1. Reads pre-computed features from offline store (Parquet files)
2. Pushes features to online store (SQLite) for fast retrieval
3. Makes features available for `get_online_features()` (inference)

### Step 4: Run Demo

```bash
python demo.py
```

The demo demonstrates:
1. Feature discovery & metadata exploration
2. Offline feature retrieval (pre-computed + on-demand features)
3. On-demand transformations applied at retrieval time
4. Feature statistics and validation



### Step 5: Run Streamlit App

```bash
streamlit run app.py
```

Interactive web application for:
- Single store-department predictions
- Batch predictions
- Feature exploration
- Model information


