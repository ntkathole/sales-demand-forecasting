"""
Data Download and Preparation Script for Sales Demand Forecasting

This script downloads the REAL Walmart sales dataset from Kaggle and prepares it for Feast.

The Walmart dataset (from Kaggle competition: walmart-recruiting-store-sales-forecasting) contains:

Files:
- train.csv: 421,570 records with Store, Dept, Date, Weekly_Sales, IsHoliday
- features.csv: Store-level features with Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment  
- stores.csv: Store metadata with Type (A/B/C) and Size

Dataset Details:
- 45 stores across different regions
- 99 departments per store (1-99, but not all stores have all departments)
- Sales data from 2010-02-05 to 2012-11-01
- 143 weeks of historical data

Usage:
    python download_data.py

Requirements:
- Kaggle API credentials configured
- Accept competition rules at: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/rules
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import zipfile


def create_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path(__file__).parent / "feature_repo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_from_kaggle(data_dir):
    """
    Download actual Walmart dataset from Kaggle.
    Requires Kaggle API credentials.
    
    Setup:
    1. Create Kaggle account at https://www.kaggle.com
    2. Go to Account settings > API > Create New API Token
    3. Place kaggle.json in ~/.kaggle/ directory
    4. Install: pip install kaggle
    5. Accept competition rules at the competition page
    """
    print("="*70)
    print("DOWNLOADING REAL WALMART DATASET FROM KAGGLE")
    print("="*70)
    print()
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("✅ Kaggle package found")
        print("📥 Downloading dataset...")
        print()
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        competition = 'walmart-recruiting-store-sales-forecasting'
        print(f"Competition: {competition}")
        print(f"Destination: {data_dir}")
        print()
        
        # Download files
        api.competition_download_files(competition, path=str(data_dir), quiet=False)
        
        # Extract zip files
        print("\n📦 Extracting files...")
        
        # First, extract the main competition zip file
        main_zip = data_dir / f'{competition}.zip'
        if main_zip.exists():
            print(f"  Extracting main competition archive...")
            try:
                with zipfile.ZipFile(main_zip, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"    ✓ Extracted competition files")
            except Exception as e:
                print(f"    ✗ Error extracting main archive: {e}")
        
        # Now extract individual CSV zip files that need extraction
        csv_zips = {
            'train.csv.zip': 'train.csv',
            'features.csv.zip': 'features.csv'
        }
        
        extracted_files = []
        
        # Extract zipped CSV files
        for zip_name, csv_name in csv_zips.items():
            zip_path = data_dir / zip_name
            csv_path = data_dir / csv_name
            
            # Check if CSV already exists
            if csv_path.exists():
                print(f"  ✓ {csv_name} already exists")
                extracted_files.append(csv_name)
                continue
            
            # Extract if zip exists
            if zip_path.exists():
                print(f"  Extracting {zip_name}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print(f"    ✓ Extracted {csv_name}")
                    extracted_files.append(csv_name)
                    # Optional: remove zip after extraction to save space
                    # zip_path.unlink()
                except Exception as e:
                    print(f"    ✗ Error extracting {zip_name}: {e}")
            else:
                print(f"  ✗ {zip_name} not found")
        
        # Check for stores.csv (not zipped in main archive)
        stores_csv = data_dir / 'stores.csv'
        if stores_csv.exists():
            print(f"  ✓ stores.csv found")
            extracted_files.append('stores.csv')
        
        # List extracted CSV files
        print("\n📂 Extracted CSV files:")
        for csv_file in extracted_files:
            csv_path = data_dir / csv_file
            if csv_path.exists():
                size_mb = csv_path.stat().st_size / 1024 / 1024
                print(f"  ✓ {csv_file} ({size_mb:.1f} MB)")
        
        # Verify required files exist
        required_files = ['train.csv', 'features.csv', 'stores.csv']
        missing_files = [f for f in required_files if not (data_dir / f).exists()]
        
        if missing_files:
            print(f"\n❌ Missing required files: {missing_files}")
            print("\nAvailable files:")
            for f in data_dir.glob("*.csv"):
                print(f"  - {f.name}")
            print("\nPlease check:")
            print("  1. Competition rules are accepted")
            print("  2. You have access to the competition data")
            return False
        
        print("\n✅ Dataset downloaded successfully!")
        print(f"   Files: {', '.join(required_files)}")
        return True
            
    except ImportError:
        print("❌ Kaggle package not installed")
        print("   Install with: pip install kaggle")
        print()
        print("   Then setup API credentials:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Create API Token")
        print("   3. Place kaggle.json in ~/.kaggle/")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print()
        print("Common issues:")
        print("  - Kaggle API credentials not configured")
        print("  - Need to accept competition rules at:")
        print(f"    https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/rules")
        return False


def load_and_explore_data(data_dir):
    """Load the Walmart CSV files and display information."""
    print("\n" + "="*70)
    print("LOADING AND EXPLORING WALMART DATASET")
    print("="*70)
    print()
    
    # Load CSV files
    print("📂 Loading CSV files...")
    try:
        train_df = pd.read_csv(data_dir / 'train.csv')
        features_df = pd.read_csv(data_dir / 'features.csv')
        stores_df = pd.read_csv(data_dir / 'stores.csv')
        
        print(f"✅ train.csv: {len(train_df):,} records")
        print(f"✅ features.csv: {len(features_df):,} records")
        print(f"✅ stores.csv: {len(stores_df):,} records")
        print()
        
        # Display dataset info
        print("📊 Dataset Summary:")
        print(f"   Stores: {train_df['Store'].nunique()}")
        print(f"   Departments: {train_df['Dept'].nunique()}")
        print(f"   Date Range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"   Total Weekly Sales Records: {len(train_df):,}")
        print()
        
        print("🏪 Store Types:")
        print(stores_df['Type'].value_counts().to_string())
        print()
        
        print("📋 train.csv columns:", list(train_df.columns))
        print("📋 features.csv columns:", list(features_df.columns))
        print("📋 stores.csv columns:", list(stores_df.columns))
        print()
        
        # Sample data
        print("👀 Sample from train.csv:")
        print(train_df.head())
        print()
        
        return train_df, features_df, stores_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required CSV file not found: {e}")
        print("   Please run download first.")
        sys.exit(1)


def create_feature_datasets(train_df, features_df, stores_df, data_dir):
    """
    Transform raw Walmart data into feature datasets for Feast.
    
    This creates parquet files optimized for Feast feature views:
    - sales_features.parquet: Historical sales with time series features
    - store_features.parquet: Store-level external factors
    - store_metadata.parquet: Store characteristics
    """
    print("="*70)
    print("CREATING FEAST FEATURE DATASETS")
    print("="*70)
    print()
    
    # Convert dates
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Create composite entity key: store_dept
    train_df['store_dept'] = train_df['Store'].astype(str) + '_' + train_df['Dept'].astype(str)
    
    # ===== 1. SALES HISTORY FEATURES =====
    print("📊 Creating sales history features...")
    
    # Sort for time series operations
    train_df = train_df.sort_values(['store_dept', 'Date'])
    
    # Calculate time series features
    sales_features = []
    for store_dept in train_df['store_dept'].unique():
        dept_df = train_df[train_df['store_dept'] == store_dept].copy()
        
        # Lag features (previous weeks)
        dept_df['sales_lag_1'] = dept_df['Weekly_Sales'].shift(1)
        dept_df['sales_lag_2'] = dept_df['Weekly_Sales'].shift(2)
        dept_df['sales_lag_4'] = dept_df['Weekly_Sales'].shift(4)
        
        # Rolling statistics (moving averages)
        dept_df['sales_rolling_mean_4'] = dept_df['Weekly_Sales'].rolling(window=4, min_periods=1).mean()
        dept_df['sales_rolling_mean_12'] = dept_df['Weekly_Sales'].rolling(window=12, min_periods=1).mean()
        dept_df['sales_rolling_std_4'] = dept_df['Weekly_Sales'].rolling(window=4, min_periods=1).std().fillna(0)
        
        # Trend indicator
        dept_df['sales_trend'] = dept_df['Weekly_Sales'] - dept_df['sales_rolling_mean_4']
        
        sales_features.append(dept_df)
    
    sales_df = pd.concat(sales_features, ignore_index=True)
    
    # Fill NaN values from shift operations
    sales_df = sales_df.fillna(0)
    
    # Select and rename columns for Feast
    sales_feast_df = sales_df[[
        'store_dept', 'Date', 'Weekly_Sales', 'IsHoliday',
        'sales_lag_1', 'sales_lag_2', 'sales_lag_4',
        'sales_rolling_mean_4', 'sales_rolling_mean_12', 
        'sales_rolling_std_4', 'sales_trend'
    ]].copy()
    
    sales_feast_df = sales_feast_df.rename(columns={'Date': 'date', 'Weekly_Sales': 'weekly_sales', 'IsHoliday': 'is_holiday'})
    
    # Save to parquet
    output_path = data_dir / 'sales_features.parquet'
    sales_feast_df.to_parquet(output_path, index=False)
    print(f"   ✅ Saved {len(sales_feast_df):,} records to sales_features.parquet")
    
    # ===== 2. STORE-LEVEL EXTERNAL FEATURES =====
    print("🌡️  Creating store-level external features...")
    
    # Merge features with stores metadata
    store_features_df = features_df.merge(stores_df, on='Store', how='left')
    
    # Create store_dept combinations (broadcast store features to all depts)
    # Get unique store_dept from training data
    unique_store_depts = train_df[['Store', 'Dept', 'store_dept']].drop_duplicates()
    
    # Expand store features to store_dept level
    store_expanded = []
    for _, row in unique_store_depts.iterrows():
        store_data = store_features_df[store_features_df['Store'] == row['Store']].copy()
        store_data['store_dept'] = row['store_dept']
        store_expanded.append(store_data)
    
    store_expanded_df = pd.concat(store_expanded, ignore_index=True)
    
    # Clean and prepare features
    store_expanded_df['Date'] = pd.to_datetime(store_expanded_df['Date'])
    
    # Fill missing markdown values with 0
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        store_expanded_df[col] = store_expanded_df[col].fillna(0)
    
    # Calculate total markdown
    store_expanded_df['total_markdown'] = store_expanded_df[markdown_cols].sum(axis=1)
    store_expanded_df['has_markdown'] = (store_expanded_df['total_markdown'] > 0).astype(int)
    
    # Select columns for Feast
    store_feast_df = store_expanded_df[[
        'store_dept', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'total_markdown', 'has_markdown', 'Type', 'Size'
    ]].copy()
    
    # Rename columns to lowercase
    store_feast_df = store_feast_df.rename(columns={
        'Date': 'date',
        'Temperature': 'temperature',
        'Fuel_Price': 'fuel_price',
        'CPI': 'cpi',
        'Unemployment': 'unemployment',
        'MarkDown1': 'markdown1',
        'MarkDown2': 'markdown2',
        'MarkDown3': 'markdown3',
        'MarkDown4': 'markdown4',
        'MarkDown5': 'markdown5',
        'Type': 'store_type',
        'Size': 'store_size'
    })
    
    # Save to parquet
    output_path = data_dir / 'store_features.parquet'
    store_feast_df.to_parquet(output_path, index=False)
    print(f"   ✅ Saved {len(store_feast_df):,} records to store_features.parquet")
    
    print()
    print("="*70)
    print("✅ FEATURE DATASETS CREATED SUCCESSFULLY!")
    print("="*70)
    print()
    print("Files created:")
    print(f"  📁 {data_dir}/sales_features.parquet")
    print(f"  📁 {data_dir}/store_features.parquet")
    print()
    print("Feature datasets are ready for Feast!")
    print()
    print("Next steps:")
    print("  1. cd feature_repo")
    print("  2. feast apply")
    print("  3. cd ..")
    print("  4. python demo.py")
    print()


def main():
    print("="*70)
    print("WALMART SALES FORECASTING - DATA PREPARATION")
    print("="*70)
    print()
    
    # Create data directory
    data_dir = create_data_directory()
    print(f"📂 Data directory: {data_dir}")
    print()
    
    # Check if data already exists
    required_files = ['train.csv', 'features.csv', 'stores.csv']
    data_exists = all((data_dir / f).exists() for f in required_files)
    
    if data_exists:
        print("✅ Walmart dataset already exists")
        print("   Skipping download...")
        print()
    else:
        # Download from Kaggle
        success = download_from_kaggle(data_dir)
        if not success:
            print("\n❌ Failed to download dataset")
            print("\nPlease ensure:")
            print("  1. Kaggle package is installed: pip install kaggle")
            print("  2. API credentials are configured (~/.kaggle/kaggle.json)")
            print("  3. Competition rules accepted at:")
            print("     https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/rules")
            sys.exit(1)
    
    # Load and explore data
    train_df, features_df, stores_df = load_and_explore_data(data_dir)
    
    # Create feature datasets for Feast
    create_feature_datasets(train_df, features_df, stores_df, data_dir)
    
    print("="*70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
