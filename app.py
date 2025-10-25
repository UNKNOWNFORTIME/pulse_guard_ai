import streamlit as st
import pandas as pd
import joblib
import traceback
import plotly.express as px  # For the pie chart
import io  # For the download button

# --- 1. Load Your Model ---
try:
    model = joblib.load("transformer_failure_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Define Feature List ---
# ⚠️ This MUST match the columns your model was trained on
FEATURE_COLUMNS = [
    "LOCATION", 
    "POWER", 
    "SELF_PROTECTION", 
    "Average_earth_discharge_density_DDT_Rays_km_2_a_o", 
    "Maximum_ground_discharge_density_DDT_Rays_km_2_a_o", 
    "Burning_rate_Failures_year", 
    "Criticality_according_to_previous_study_for_ceramics_level", 
    "Removable_connectors", 
    "Type_of_clients", 
    "Number_of_users", 
    "Electric_power_not_supplied_EENS_kWh", 
    "Type_of_installation", 
    "Air_network", 
    "Circuit_Queue", 
    "km_of_network_LT"]

# --- 3. Download Button Helper Function ---
# This converts the DataFrame to CSV in memory
@st.cache_data
def convert_df_to_csv(df):
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    return output.getvalue()

# --- 4. Page Configuration ---
st.set_page_config(page_title="Transformer Failure Prediction", page_icon="⚡", layout="wide")
st.title("⚡ Transformer Failure Prediction Dashboard")

# --- 5. Sidebar - Mode Selection ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Choose your mode:",
    ["Single Prediction (Manual Entry)", "Batch Upload & Analyze"]
)

# --- 6. Mode 1: Single Prediction ---
if app_mode == "Single Prediction (Manual Entry)":
    st.sidebar.header("Input Features")
    st.subheader("Enter data for a single transformer:")
    
    # Create a dictionary to hold user inputs
    input_data = {}
    
    # We'll use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    # ⚠️ Update these with your *actual* feature names and sensible defaults
    # I am using a few examples. Add ALL features from your FEATURE_COLUMNS list.
    with col1:
        input_data["LOCATION"] = st.number_input("Location (Code)", value=100)
        input_data["POWER"] = st.number_input("Power (kVA)", value=160.0, step=10.0)
        input_data["SELF_PROTECTION"] = st.selectbox("Self Protection (1=Yes, 0=No)", [0, 1])
        input_data["Type_of_clients"] = st.number_input("Type of Clients", value=1)
        input_data["Number_of_users"] = st.number_input("Number of Users", value=50, step=1)
        input_data["km_of_network_LT"] = st.number_input("km of LT Network", value=1.5, step=0.1)

    with col2:
        # Add the rest of your features here...
        # Example for the complex name:
        input_data["Average earth discharge density DDT [Rays/km^2-a駉]"] = st.number_input("Avg Earth Discharge", value=2.5, step=0.1)
        # ... add all other features from FEATURE_COLUMNS ...
        # For columns not added, we'll use 0 as a default
        for col in FEATURE_COLUMNS:
            if col not in input_data:
                input_data[col] = 0
    
    # Create a DataFrame from the single input
    input_df = pd.DataFrame([input_data])
    
    if st.button("Predict Failure", type="primary", use_container_width=True):
        try:
            # Predict
            prediction = model.predict(input_df[FEATURE_COLUMNS])[0]
            prediction_proba = model.predict_proba(input_df[FEATURE_COLUMNS])[0]

            # Display Result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"Prediction: **Failure Likely** (Class {prediction})")
                st.write(f"Confidence: {prediction_proba[1] * 100:.2f}%")
            else:
                st.success(f"Prediction: **No Failure Detected** (Class {prediction})")
                st.write(f"Confidence: {prediction_proba[0] * 100:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred: {traceback.format_exc()}")

# --- 7. Mode 2: Batch Upload & Analyze ---
elif app_mode == "Batch Upload & Analyze":
    st.subheader("Upload your Transformer Data (CSV)")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with your transformer data.",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            # --- Load Data ---
            data = pd.read_csv(uploaded_file)
            st.write("Data loaded successfully. Here's a preview:")
            st.dataframe(data.head())
            
            # --- Clean Data (Same as tryinnff.py) ---
            st.write("Cleaning data... (converting text to 0, filling missing values)")
            
            # Check for missing columns
            original_cols = set(data.columns)
            missing_cols = [col for col in FEATURE_COLUMNS if col not in original_cols]
            
            if missing_cols:
                st.warning(f"Warning: The following columns were not in your CSV and will be filled with 0: {', '.join(missing_cols)}")
                for col in missing_cols:
                    data[col] = 0 # Add missing columns and fill with 0
            
            # Force-clean all feature columns
            for col in FEATURE_COLUMNS:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data_cleaned = data.fillna(0)
            
            # --- Make Predictions ---
            st.write("Running predictions on all rows...")
            X_new = data_cleaned[FEATURE_COLUMNS]
            data['prediction'] = model.predict(X_new)
            data['failure_probability_%'] = model.predict_proba(X_new)[:, 1] * 100
            
            st.success("Predictions complete! Here's the analysis:")
            
            # --- 8. Display Dashboard Features ---
            
            # Get counts
            prediction_counts = data['prediction'].value_counts()
            healthy_count = prediction_counts.get(0, 0)
            failure_count = prediction_counts.get(1, 0)

            # --- Metrics & Pie Chart (in columns) ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Breakdown")
                st.metric(label="✅ Healthy Transformers", value=healthy_count)
                st.metric(label="🔥 Failures Predicted", value=failure_count)

            with col2:
                st.subheader("Result Distribution")
                if healthy_count + failure_count > 0:
                    pie_data = pd.DataFrame({
                        'Category': ['Healthy (0)', 'Failure (1)'],
                        'Count': [healthy_count, failure_count]
                    })
                    fig = px.pie(
                        pie_data, 
                        values='Count', 
                        names='Category', 
                        title='Pie Chart: Prediction Breakdown',
                        color_discrete_map={'Healthy (0)': 'green', 'Failure (1)': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No data to display in chart.")

            # --- Full Data Table & Download ---
            st.subheader("Prediction Results (Full Data)")
            st.dataframe(data)
            
            csv_data = convert_df_to_csv(data)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_data,
                file_name="transformer_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"An error occurred during batch processing: {traceback.format_exc()}")