"""
Sales Demand Forecasting with Feature Store - Demo Script

This script demonstrates key Feature Store operations for ML feature serving:

Operations:
1. Feature discovery and metadata exploration
2. Offline feature retrieval for model training (with point-in-time correctness)
3. Feature materialization (offline → online store)
4. Online feature serving (low-latency lookup)
5. Training-serving consistency

Feature Engineering Pattern:
- Time-series features pre-computed in ETL pipeline (download_data.py)
- Feature Store serves features consistently across training and inference
- On-demand transformations applied at retrieval time

Prerequisites:
- Run `python download_data.py` to compute and prepare features
- Run `cd feature_repo && feast apply` to register features in Feature Store
"""

import pandas as pd
from datetime import timedelta
from pathlib import Path
import time

from feast import FeatureStore

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")


def main():
    print_section("WALMART SALES FORECASTING WITH FEATURE STORE")
    
    print("🚀 Initializing Feast Feature Store...")
    
    try:
        store = FeatureStore(repo_path="feature_repo")
        print(f"✅ Connected to Feast project: {store.project}")
        print(f"   Registry: {store.config.registry.path}")
        print(f"   Offline Store: {store.config.offline_store.type}")
        print(f"   Online Store: {store.config.online_store.type}")
    except Exception as e:
        print(f"Failed to initialize Feast: {e}")
        print("\n📖 Make sure you've run:")
        print("   1. python download_data.py")
        print("   2. cd feature_repo && feast apply")
        return
    
    print_section("OFFLINE FEATURE RETRIEVAL (Training)")
    
    print("📊 Creating entity DataFrame for feature retrieval...")
    print("   Loading all available sales data...")
    
    data_path = Path("feature_repo/data/sales_features.parquet")
    if not data_path.exists():
        print("❌ Sales data not found. Please run: python download_data.py")
        return
    
    raw_df = pd.read_parquet(data_path)
    
    # Use ALL data (not just a sample)
    entity_df = raw_df[['store', 'dept', 'date']].copy()
    entity_df = entity_df.rename(columns={'date': 'event_timestamp'})
    
    print(f"   ✅ Entity DataFrame: {len(entity_df):,} rows")
    print(f"   📍 Store-Dept combinations: {len(entity_df[['store', 'dept']].drop_duplicates()):,}")
    print(f"   📅 Date range: {entity_df['event_timestamp'].min().date()} to {entity_df['event_timestamp'].max().date()}")
    print(f"   💾 Dataset size: ~{len(entity_df) / 1000:.1f}K records")
    
    print("\n🔄 Retrieving features from Feature Store offline store...")
    print("   Including pre-computed time-series features and on-demand transformations...")
    
    start_time = time.time()
    
    try:
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service("demand_forecasting_service"),
        ).to_df()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Features retrieved in {elapsed_time:.2f} seconds!")
        print(f"   Shape: {training_df.shape}")
        print(f"   Total features: {training_df.shape[1]}")
        
    except Exception as e:
        print(f"\nFeature retrieval failed: {e}")
        return
    
    # Display features
    print("\n📋 Retrieved Features:")
    print(f"\n🔍 ALL COLUMNS: {list(training_df.columns)}")
    print(f"\n🔍 COLUMN DTYPES:\n{training_df.dtypes}")
    
    feature_cols = [col for col in training_df.columns if col not in ['store', 'dept', 'event_timestamp']]
    
    base_features = [col for col in feature_cols if not any(x in col for x in 
                    ['lag', 'rolling', 'trend', 'normalized', 'velocity', 'acceleration', 'stability', 
                     'per_sqft', 'efficiency'])]
    time_series = [col for col in feature_cols if any(x in col for x in ['lag', 'rolling'])]
    odfv_features = [col for col in feature_cols if col not in base_features and col not in time_series]
    
    print(f"\n   📊 BASE Features ({len(base_features)} features):")
    for col in base_features:
        print(f"      • {col}")
    
    print(f"\n   ⏰ TIME-SERIES Features ({len(time_series)} features - pre-computed):")
    for col in time_series:
        value_sample = training_df[col].iloc[0] if len(training_df) > 0 else 'N/A'
        print(f"      • {col} (example: {value_sample:.2f})")
    
    print(f"\n   ✨ ON-DEMAND Features ({len(odfv_features)} features):")
    for col in odfv_features:
        print(f"      • {col}")
    
    # Sample data
    print(f"\n📊 Sample Data (first 3 rows, key columns):")
    display_cols = ['store', 'dept', 'weekly_sales', 'sales_lag_1', 'sales_rolling_mean_4', 
                    'sales_rolling_std_4']
    available_cols = [col for col in display_cols if col in training_df.columns]
    print(training_df[available_cols].head(3).to_string(index=False))
    
    # Verification
    print("\n✅ VERIFICATION: Feature Statistics")
    
    if 'sales_lag_1' in training_df.columns:
        non_null_lags = training_df['sales_lag_1'].notna().sum()
        print(f"   ✅ sales_lag_1: {non_null_lags}/{len(training_df)} non-null values")
    
    if 'sales_rolling_mean_4' in training_df.columns:
        non_null_rolling = training_df['sales_rolling_mean_4'].notna().sum()
        print(f"   ✅ sales_rolling_mean_4: {non_null_rolling}/{len(training_df)} non-null values")
    
    if 'sales_velocity' in training_df.columns:
        non_null_velocity = training_df['sales_velocity'].notna().sum()
        print(f"   ✅ sales_velocity (on-demand): {non_null_velocity}/{len(training_df)} non-null values")
    


if __name__ == "__main__":
    main()

