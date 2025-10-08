"""
Sales Demand Forecasting - Interactive Streamlit App

This app demonstrates real-time sales forecasting using:
- Feature Store for feature serving
- LightGBM for predictions
- Interactive UI for model inference

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from feast import FeatureStore
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

# Page configuration
st.set_page_config(
    page_title="Sales Demand Forecasting",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feature-value {
        font-family: monospace;
        background-color: #e8f4f8;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_feast_store():
    """Load and cache Feast Feature Store."""
    try:
        store = FeatureStore(repo_path="feature_repo")
        store.refresh_registry()
        return store
    except Exception as e:
        st.error(f"Error loading Feast store: {e}")
        return None


@st.cache_resource
def load_or_train_model(_store):
    """Load existing model or train a new one."""
    model_path = Path("model/sales_forecasting_model.pkl")
    
    if model_path.exists():
        model_data = joblib.load(model_path)
        # Handle both old format (just model) and new format (dict with model + features)
        if isinstance(model_data, dict):
            return model_data, "loaded"
        else:
            # Old format - just the model
            return {'model': model_data, 'feature_cols': None}, "loaded"
    
    # Train new model
    st.info("📊 Training new model... This will take a moment.")
    
    # Load training data
    data_path = Path("feature_repo/data/sales_features.parquet")
    if not data_path.exists():
        st.error("Training data not found. Please run download_data.py first.")
        return None, "error"
    
    raw_df = pd.read_parquet(data_path)
    
    # Create entity DataFrame (using separate store and dept columns)
    unique_combinations = raw_df[['store', 'dept']].drop_duplicates().head(100)
    sample_df = raw_df.merge(unique_combinations, on=['store', 'dept'], how='inner')
    sample_df = sample_df.sort_values('date')
    cutoff_date = sample_df['date'].max() - timedelta(weeks=20)
    sample_df = sample_df[sample_df['date'] >= cutoff_date]
    
    entity_df = sample_df[['store', 'dept', 'date']].copy()
    entity_df = entity_df.rename(columns={'date': 'event_timestamp'})
    
    # Retrieve features
    training_df = _store.get_historical_features(
        entity_df=entity_df,
        features=_store.get_feature_service("demand_forecasting_service"),
    ).to_df()
    
    # Prepare features
    training_df = training_df[training_df['weekly_sales'] > 0].copy()
    
    # Base features + external factors + on-demand transformations
    # Check which columns are actually available
    available_cols = set(training_df.columns)
    
    feature_cols = [
        # Base features
        'is_holiday',
        # External factors
        'temperature', 'fuel_price', 'cpi', 'unemployment',
        'total_markdown', 'has_markdown', 'store_size',
        # On-demand transformations
        'sales_normalized', 'temperature_normalized',
        'sales_per_sqft', 'markdown_efficiency'
    ]
    
    # Add pre-computed time-series features if available
    time_series_features = ['sales_lag_1', 'sales_lag_2', 'sales_lag_4',
                            'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_4']
    feature_cols.extend([f for f in time_series_features if f in available_cols])
    
    training_df['store_type_encoded'] = training_df['store_type'].astype('category').cat.codes
    feature_cols.append('store_type_encoded')
    
    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in available_cols]
    
    print(f"📊 Using {len(feature_cols)} features for training:")
    print(f"   Available: {', '.join(feature_cols)}")
    
    training_df = training_df.dropna(subset=feature_cols + ['weekly_sales'])
    
    X = training_df[feature_cols]
    y = training_df['weekly_sales']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    
    # Save model and feature columns
    model_path.parent.mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, model_path)
    
    return model_data, "trained"


@st.cache_data
def load_available_store_depts():
    """Load available store-department combinations."""
    data_path = Path("feature_repo/data/sales_features.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        # Create list of (store, dept) tuples
        store_depts = df[['store', 'dept']].drop_duplicates().sort_values(['store', 'dept'])
        # Format as "Store X, Dept Y" for display
        store_depts['label'] = store_depts.apply(lambda x: f"Store {x['store']}, Dept {x['dept']}", axis=1)
        return store_depts[['store', 'dept', 'label']].to_dict('records')
    return []


def get_online_features(store, store_dept_list, feature_service_name="demand_forecasting_service"):
    """Retrieve features from Feast online store.
    
    Args:
        store_dept_list: List of dicts with 'store' and 'dept' keys
    """
    try:
        # Create entity rows with separate store and dept
        entity_rows = [{"store": sd['store'], "dept": sd['dept']} for sd in store_dept_list]
        
        # Get online features
        feature_vector = store.get_online_features(
            features=store.get_feature_service(feature_service_name),
            entity_rows=entity_rows
        ).to_dict()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_vector)
        return features_df
    
    except Exception as e:
        st.error(f"Error retrieving features: {e}")
        st.info("💡 Tip: Make sure features are materialized. Run: `cd feature_repo && feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)`")
        return None


def predict_sales(model_data, features_df):
    """Make predictions using the model.
    
    Args:
        model_data: Dict with 'model' and 'feature_cols' keys
        features_df: DataFrame with features
    """
    model = model_data['model']
    feature_cols = model_data.get('feature_cols')
    
    # Encode store type (needed for both training and prediction)
    if 'store_type' in features_df.columns:
        features_df = features_df.copy()
        features_df['store_type_encoded'] = features_df['store_type'].astype('category').cat.codes
    
    # If feature_cols not stored (old model), infer from available columns
    if feature_cols is None:
        available_cols = set(features_df.columns)
        feature_cols = [
            'is_holiday', 'temperature', 'fuel_price', 'cpi', 'unemployment',
            'total_markdown', 'has_markdown', 'store_size',
            'sales_normalized', 'temperature_normalized',
            'sales_per_sqft', 'markdown_efficiency'
        ]
        time_series_features = ['sales_lag_1', 'sales_lag_2', 'sales_lag_4',
                                'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_4']
        feature_cols.extend([f for f in time_series_features if f in available_cols])
        if 'store_type_encoded' in available_cols:
            feature_cols.append('store_type_encoded')
        feature_cols = [c for c in feature_cols if c in available_cols]
    
    # Use exact feature columns from training (IMPORTANT: same order, same count)
    # Filter to only columns that exist in features_df
    available_feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(f"⚠️  Warning: Missing features: {missing}")
    
    # Convert all feature columns to numeric (Feast online store returns objects)
    X = features_df[available_feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN values with 0
    X = X.fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions


def create_prediction_chart(predictions_df):
    """Create interactive prediction visualization."""
    fig = go.Figure()
    
    # Create labels for x-axis
    predictions_df['label'] = predictions_df.apply(
        lambda x: f"Store {x['store']}, Dept {x['dept']}", axis=1
    )
    
    fig.add_trace(go.Bar(
        x=predictions_df['label'],
        y=predictions_df['predicted_sales'],
        name='Predicted Sales',
        marker_color='#1f77b4',
        text=predictions_df['predicted_sales'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Sales Forecast by Store-Department',
        xaxis_title='Store-Department',
        yaxis_title='Predicted Weekly Sales ($)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_feature_importance_chart(model):
    """Create feature importance visualization."""
    importance = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    fig = px.bar(
        importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🏪 Sales Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown("**Powered by Feast Feature Store & LightGBM**")
    
    # Setup info banner
    with st.expander("📋 Setup Instructions", expanded=False):
        st.markdown("""
        **Before using this app, make sure you've completed these steps:**
        
        1. **Download Data & Compute Features:**
           ```bash
           python download_data.py
           ```
        
        2. **Register Features with Feast:**
           ```bash
           cd feature_repo && feast apply
           ```
        
        3. **Materialize Features to Online Store:**
           ```bash
           cd feature_repo
           feast materialize-incremental $(date -u +'%Y-%m-%dT%H:%M:%S')
           ```
           
        💡 **Note:** If you see "N/A" values in predictions, it means features haven't been materialized yet. Run step 3 above.
        """)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load Feast store
        with st.spinner("Loading Feast Feature Store..."):
            store = load_feast_store()
        
        if store:
            st.success("✅ Feast Connected")
            st.info(f"**Project:** {store.project}")
        else:
            st.error("❌ Feast Not Available")
            st.stop()
        
        # Load model
        with st.spinner("Loading ML Model..."):
            model_data, status = load_or_train_model(store)
        
        if model_data:
            if status == "loaded":
                st.success("✅ Model Loaded")
            else:
                st.success("✅ Model Trained")
        else:
            st.error("❌ Model Not Available")
            st.stop()
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "**Select Mode**",
            ["Single Prediction", "Batch Prediction", "Feature Explorer", "Model Info"],
            index=0
        )
    
    # Main content based on mode
    if mode == "Single Prediction":
        show_single_prediction(store, model_data)
    
    elif mode == "Batch Prediction":
        show_batch_prediction(store, model_data)
    
    elif mode == "Feature Explorer":
        show_feature_explorer(store)
    
    elif mode == "Model Info":
        show_model_info(model_data)


def show_single_prediction(store, model_data):
    """Single prediction mode."""
    st.header("🎯 Single Store-Department Prediction")
    st.markdown("Select a store-department combination to get a sales forecast.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load available store-depts
        store_depts = load_available_store_depts()
        
        if not store_depts:
            st.error("No store-department data available.")
            return
        
        selected_store_dept = st.selectbox(
            "Select Store-Department",
            options=range(len(store_depts)),
            format_func=lambda i: store_depts[i]['label'],
            index=0
        )
        
        if st.button("🔮 Predict Sales", type="primary", use_container_width=True):
            with st.spinner("Retrieving features from Feast and predicting..."):
                # Get features from Feast online store
                features_df = get_online_features(store, [store_depts[selected_store_dept]])
                
                if features_df is not None and len(features_df) > 0:
                    # Make prediction
                    prediction = predict_sales(model_data, features_df)[0]
                    
                    # Display prediction
                    st.markdown("### 📊 Prediction Result")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Store-Department", store_depts[selected_store_dept]['label'])
                    
                    with metric_col2:
                        st.metric("Predicted Weekly Sales", f"${prediction:,.2f}")
                    
                    with metric_col3:
                        monthly_estimate = prediction * 4.33
                        st.metric("Estimated Monthly Sales", f"${monthly_estimate:,.2f}")
                    
                    # Display features used
                    st.markdown("### 🔍 Features Used for Prediction")
                    
                    # Helper function to safely format values (handles NaN/None)
                    def safe_format(value, format_str=",.2f", prefix="$", suffix="", na_text="N/A"):
                        try:
                            if pd.isna(value) or value is None:
                                return na_text
                            if prefix or suffix:
                                return f"{prefix}{value:{format_str}}{suffix}"
                            return f"{value:{format_str}}"
                        except:
                            return na_text
                    
                    # Build feature display dict with conditional inclusion
                    feature_display = {}
                    
                    # Historical Sales (only if features exist)
                    if all(col in features_df.columns for col in ['sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_rolling_mean_4']):
                        feature_display["Historical Sales"] = {
                            "Last Week Sales": safe_format(features_df['sales_lag_1'].iloc[0]),
                            "2 Weeks Ago": safe_format(features_df['sales_lag_2'].iloc[0]),
                            "4 Weeks Ago": safe_format(features_df['sales_lag_4'].iloc[0]),
                            "4-Week Average": safe_format(features_df['sales_rolling_mean_4'].iloc[0]),
                        }
                    
                    # External Factors
                    if all(col in features_df.columns for col in ['temperature', 'fuel_price', 'cpi', 'unemployment']):
                        feature_display["External Factors"] = {
                            "Temperature": safe_format(features_df['temperature'].iloc[0], ".1f", "", "°F"),
                            "Fuel Price": safe_format(features_df['fuel_price'].iloc[0], ".2f", "$", "/gal"),
                            "CPI": safe_format(features_df['cpi'].iloc[0], ".2f", "", ""),
                            "Unemployment": safe_format(features_df['unemployment'].iloc[0], ".2f", "", "%"),
                        }
                    
                    # Promotions
                    if all(col in features_df.columns for col in ['total_markdown', 'has_markdown', 'is_holiday']):
                        feature_display["Promotions"] = {
                            "Total Markdown": safe_format(features_df['total_markdown'].iloc[0]),
                            "Has Markdown": "Yes" if features_df['has_markdown'].iloc[0] else "No",
                            "Holiday Week": "Yes" if features_df['is_holiday'].iloc[0] else "No",
                        }
                    
                    # Store Characteristics
                    if all(col in features_df.columns for col in ['store_type', 'store_size']):
                        feature_display["Store Characteristics"] = {
                            "Store Type": str(features_df['store_type'].iloc[0]) if pd.notna(features_df['store_type'].iloc[0]) else "N/A",
                            "Store Size": safe_format(features_df['store_size'].iloc[0], ",.0f", "", " sq ft"),
                        }
                    
                    # Display features in columns (adapt to number of categories)
                    num_cols = len(feature_display)
                    if num_cols > 0:
                        cols = st.columns(min(num_cols, 4))
                        for idx, (category, features) in enumerate(feature_display.items()):
                            with cols[idx % len(cols)]:
                                st.markdown(f"**{category}**")
                                for feature_name, feature_value in features.items():
                                    st.markdown(f"• {feature_name}: `{feature_value}`")
                    else:
                        st.warning("⚠️ No features available to display. Make sure features are materialized to the online store.")
                    
                    # Explanation
                    st.info("💡 **How it works:** Feast retrieves the latest features from the online store in < 10ms, then the model uses these features to generate the prediction.")
    
    with col2:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app demonstrates:
        
        ✅ **Real-time Feature Serving**
        - Features retrieved from Feast online store
        
        ✅ **Low Latency**
        - Sub-10ms feature retrieval
        
        ✅ **Consistent Features**
        - Same features as training
        
        ✅ **Production Ready**
        - Scalable architecture
        """)


def show_batch_prediction(store, model_data):
    """Batch prediction mode."""
    st.header("📦 Batch Prediction")
    st.markdown("Predict sales for multiple store-departments at once.")
    
    store_depts = load_available_store_depts()
    
    if not store_depts:
        st.error("No store-department data available.")
        return
    
    num_predictions = st.slider("Number of predictions", 5, 50, 10)
    
    selected_indices = st.multiselect(
        "Select Store-Departments",
        options=range(len(store_depts)),
        format_func=lambda i: store_depts[i]['label'],
        default=list(range(min(num_predictions, len(store_depts))))
    )
    
    if st.button("🔮 Predict All", type="primary"):
        if not selected_indices:
            st.warning("Please select at least one store-department.")
            return
        
        selected_store_depts = [store_depts[i] for i in selected_indices]
        
        with st.spinner(f"Predicting sales for {len(selected_store_depts)} store-departments..."):
            # Get features
            features_df = get_online_features(store, selected_store_depts)
            
            if features_df is not None and len(features_df) > 0:
                # Make predictions
                predictions = predict_sales(model_data, features_df)
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'store': features_df['store'],
                    'dept': features_df['dept'],
                    'predicted_sales': predictions,
                    'store_type': features_df['store_type'],
                    'store_size': features_df['store_size'],
                    'has_markdown': features_df['has_markdown'].map({1: 'Yes', 0: 'No'}),
                    'is_holiday': features_df['is_holiday'].map({1: 'Yes', 0: 'No'})
                })
                
                results_df = results_df.sort_values('predicted_sales', ascending=False)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(results_df))
                
                with col2:
                    st.metric("Average Forecast", f"${results_df['predicted_sales'].mean():,.2f}")
                
                with col3:
                    st.metric("Highest Forecast", f"${results_df['predicted_sales'].max():,.2f}")
                
                with col4:
                    st.metric("Total Weekly Forecast", f"${results_df['predicted_sales'].sum():,.2f}")
                
                # Visualization
                st.markdown("### 📊 Predictions Visualization")
                fig = create_prediction_chart(results_df.head(20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown("### 📋 Detailed Results")
                
                # Format for display
                display_df = results_df.copy()
                display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"${x:,.2f}")
                display_df['store_size'] = display_df['store_size'].apply(lambda x: f"{x:,} sq ft")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def show_feature_explorer(store):
    """Feature explorer mode."""
    st.header("🔍 Feature Explorer")
    st.markdown("Explore feature definitions and metadata from Feast registry.")
    
    tab1, tab2, tab3 = st.tabs(["Feature Views", "Feature Services", "Entities"])
    
    with tab1:
        st.markdown("### 📊 Feature Views")
        
        feature_views = store.list_feature_views()
        
        for fv in feature_views:
            with st.expander(f"🔹 **{fv.name}**"):
                st.markdown(f"**Description:** {fv.description}")
                st.markdown(f"**Owner:** {fv.tags.get('owner', 'N/A')}")
                st.markdown(f"**Priority:** {fv.tags.get('priority', 'N/A')}")
                st.markdown(f"**Online Serving:** {'✅ Enabled' if fv.online else '❌ Disabled'}")
                st.markdown(f"**TTL:** {fv.ttl}")
                
                st.markdown("**Features:**")
                
                features_data = []
                for feature in fv.features:
                    features_data.append({
                        "Name": feature.name,
                        "Type": str(feature.dtype),
                        "Description": feature.description[:80] + "..." if len(feature.description) > 80 else feature.description
                    })
                
                st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🎯 Feature Services")
        
        feature_services = store.list_feature_services()
        
        for fs in feature_services:
            with st.expander(f"🎯 **{fs.name}**"):
                st.markdown(f"**Description:** {fs.description}")
                st.markdown(f"**Use Case:** {fs.tags.get('use_case', 'N/A')}")
                st.markdown(f"**Stakeholders:** {fs.tags.get('stakeholders', 'N/A')}")
                
                st.markdown("**Included Feature Views:**")
                for fv_proj in fs.feature_view_projections:
                    st.markdown(f"- {fv_proj.name} ({len(fv_proj.features)} features)")
    
    with tab3:
        st.markdown("### 🏷️ Entities")
        
        entities = store.list_entities()
        
        for entity in entities:
            with st.expander(f"🏷️ **{entity.name}**"):
                st.markdown(f"**Description:** {entity.description}")
                st.markdown(f"**Value Type:** {entity.value_type}")
                st.markdown(f"**Owner:** {entity.tags.get('owner', 'N/A')}")
                st.markdown(f"**Domain:** {entity.tags.get('domain', 'N/A')}")


def show_model_info(model_data):
    """Model information mode."""
    st.header("🤖 Model Information")
    
    # Extract model from model_data dictionary
    model = model_data['model']
    feature_cols = model_data.get('feature_cols', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Details")
        st.markdown(f"""
        - **Algorithm:** LightGBM (Gradient Boosting)
        - **Number of Estimators:** {model.n_estimators}
        - **Learning Rate:** {model.learning_rate}
        - **Max Depth:** {model.max_depth}
        - **Number of Leaves:** {model.num_leaves}
        - **Number of Features:** {len(feature_cols) if feature_cols else model.n_features_}
        """)
        
        st.markdown("### 🎯 Use Case")
        st.markdown("""
        This model predicts weekly sales for Walmart store-department combinations based on:
        - Historical sales patterns
        - External factors (weather, economy)
        - Promotional activities
        - Store characteristics
        """)
    
    with col2:
        st.markdown("### 📈 Feature Importance")
        fig = create_feature_importance_chart(model)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    User Request → Streamlit UI → Feast Online Store → Feature Vector → LightGBM Model → Prediction
                                         ↓ (< 10ms)
                                    SQLite/Redis
    ```
    """)
    
    st.info("""
    💡 **Key Benefits:**
    - ✅ Real-time predictions with low latency
    - ✅ Consistent features between training and serving
    - ✅ Centralized feature management via Feast
    - ✅ Scalable architecture for production
    """)


if __name__ == "__main__":
    main()

