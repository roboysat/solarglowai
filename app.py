import streamlit as st

# Page configuration - must be the first streamlit command
st.set_page_config(
    page_title="Solar Irradiance Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from utils import load_model_files, make_prediction, save_prediction, get_prediction_history

# Custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: rgba(240, 242, 246, 0.5);
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    div.stButton > button:first-child {
        background-color: #1E88E5;
        color:white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
    }
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 5px 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("☀️ Solar Irradiance Prediction Dashboard")
st.markdown("""
This dashboard predicts solar irradiance (GHI) based on atmospheric conditions using a pre-trained ML model.
Enter the parameters below to get a prediction.
""")

# Load model and scalers
try:
    model, feature_scaler, target_scaler, model_metadata = load_model_files()
    model_loaded = True
    st.success("Model loaded successfully!")
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")
    st.info("Please ensure model files are in the correct location.")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Cloud type options
cloud_types = {
    0: "Clear",
    1: "Probably Clear",
    2: "Fog",
    3: "Water",
    4: "Super-Cooled Water",
    5: "Mixed",
    6: "Opaque Ice",
    7: "Cirrus",
    8: "Overlapping",
    9: "Overshooting",
    10: "Unknown",
    11: "Dust",
    12: "Smoke",
    15: "N/A"
}

# Input form with units displayed
with st.sidebar.form("prediction_form"):
    st.subheader("Enter Atmospheric Conditions")
    
    # Solar zenith angle
    solar_zenith = st.number_input("Solar Zenith Angle (degrees)", 
                                  min_value=0.0, 
                                  max_value=90.0, 
                                  value=45.0,
                                  step=0.1)
    
    # Temperature
    temperature = st.number_input("Temperature (°C)", 
                                 min_value=-50.0, 
                                 max_value=60.0, 
                                 value=25.0,
                                 step=0.1)
    
    # Pressure
    pressure = st.number_input("Pressure (mbar)", 
                              min_value=800.0, 
                              max_value=1200.0, 
                              value=1013.25,
                              step=0.1)
    
    # Relative Humidity
    humidity = st.number_input("Relative Humidity (%)", 
                              min_value=0.0, 
                              max_value=100.0, 
                              value=50.0,
                              step=0.1)
    
    # Wind Speed
    wind_speed = st.number_input("Wind Speed (m/s)", 
                                min_value=0.0, 
                                max_value=50.0, 
                                value=5.0,
                                step=0.1)
    
    # Cloud Type
    cloud_type = st.selectbox("Cloud Type", 
                             options=list(cloud_types.keys()),
                             format_func=lambda x: f"{x} - {cloud_types[x]}")
    
    # Current GHI (for comparison, optional)
    actual_ghi = st.number_input("Actual GHI (W/m²) - Optional", 
                                min_value=0.0, 
                                max_value=1500.0, 
                                value=0.0,
                                step=0.1)
    
    submit_button = st.form_submit_button("Predict Solar Irradiance")

# Main content area with tabs
tab1, tab2 = st.tabs(["Prediction", "Quick Statistics"])

# Prediction tab
with tab1:
    if submit_button and model_loaded:
        # Create input data dictionary
        input_data = {
            'solar_zenith': solar_zenith,
            'temp': temperature,
            'pressure': pressure,
            'rel_humidity': humidity,
            'wind_speed': wind_speed,
            'cloud_type': cloud_type,
            'actual_ghi': actual_ghi if actual_ghi > 0 else None
        }
        
        # Make prediction
        prediction, confidence = make_prediction(
            model, 
            feature_scaler, 
            target_scaler, 
            input_data
        )
        
        # Save prediction history
        save_prediction(input_data, prediction, confidence)
        
        # Display prediction with animation
        st.balloons()
        
        # Create a card-like container for results
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #1E88E5; text-align: center; margin-bottom: 20px;">Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted GHI (W/m²)", 
                value=f"{prediction:.2f}"
            )
            
        with col2:
            if actual_ghi > 0:
                difference = prediction - actual_ghi
                st.metric(
                    label="Actual GHI (W/m²)",
                    value=f"{actual_ghi:.2f}",
                    delta=f"{difference:.2f}",
                    delta_color="normal"
                )
            else:
                st.metric(
                    label="Maximum Possible GHI",
                    value="1500.00 W/m²"
                )
        
        with col3:
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%",
                delta=None
            )
            
        # Divider
        st.markdown("<hr style='margin: 30px 0'>", unsafe_allow_html=True)
        
        # Visualizations
        st.subheader("Visualization")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Create a gauge-like visualization for the prediction
            fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
            
            # Calculate the angle for the gauge (0 to 1500 W/m²)
            max_ghi = 1500
            theta = np.linspace(0, np.pi, 100)
            
            # Background gauge
            ax.plot(theta, [max_ghi] * 100, color='lightgray', linewidth=20, alpha=0.3)
            
            # Calculate the prediction percentage
            pred_percentage = min(prediction / max_ghi, 1.0)
            pred_theta = np.linspace(0, np.pi * pred_percentage, 100)
            
            # Prediction gauge
            ax.plot(pred_theta, [max_ghi] * len(pred_theta), color='#1E88E5', linewidth=20, alpha=0.8)
            
            # Add labels
            ax.set_rticks([])  # Remove radial ticks
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['0', '375', '750', '1125', '1500'])
            
            # Set y-limit and remove unwanted elements
            ax.set_ylim(0, max_ghi + 100)
            ax.spines['polar'].set_visible(False)
            ax.grid(False)
            
            # Add a title
            ax.set_title('Predicted GHI', fontsize=14, pad=20)
            
            # Add the value text in the center
            ax.text(np.pi/2, max_ghi/2, f"{prediction:.1f}\nW/m²", 
                     horizontalalignment='center', verticalalignment='center', 
                     fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
        
        with viz_col2:
            # Create a comparison chart if actual GHI is provided
            if actual_ghi > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                
                labels = ['Predicted', 'Actual']
                values = [prediction, actual_ghi]
                colors = ['#1E88E5', '#FFA726']
                
                # Calculate percentage difference
                pct_diff = abs(prediction - actual_ghi) / actual_ghi * 100 if actual_ghi > 0 else 0
                
                bars = ax.bar(labels, values, color=colors, width=0.5)
                
                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                            f'{height:.1f}', ha='center', fontsize=10)
                
                ax.set_ylabel('GHI (W/m²)')
                ax.set_title(f'Predicted vs Actual GHI\n(Difference: {pct_diff:.1f}%)', pad=10)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Set y-axis to start from 0 and go slightly above the max value
                max_val = max(values) * 1.1
                ax.set_ylim(0, max_val)
                
                st.pyplot(fig)
            else:
                # Show a chart of the input parameters and their effect
                # Create a radar chart of input parameters
                fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
                
                # Normalize input values for radar chart
                # Solar zenith: 0-90 degrees
                solar_norm = 1 - (solar_zenith / 90.0)  # Invert so lower zenith is better for irradiance
                # Temperature: -50 to 60 C
                temp_norm = (temperature + 50) / 110.0
                # Pressure: 800-1200 mbar
                pressure_norm = (pressure - 800) / 400.0
                # Humidity: 0-100%
                humidity_norm = 1 - (humidity / 100.0)  # Invert so lower humidity is better for irradiance
                # Wind speed: 0-50 m/s
                wind_norm = wind_speed / 50.0
                # Cloud type: normalized based on its effect (clear sky is best)
                cloud_impact = {
                    0: 1.0,      # Clear
                    1: 0.9,      # Probably Clear
                    2: 0.5,      # Fog
                    3: 0.6,      # Water
                    4: 0.55,     # Super-Cooled Water
                    5: 0.5,      # Mixed
                    6: 0.4,      # Opaque Ice
                    7: 0.7,      # Cirrus
                    8: 0.3,      # Overlapping
                    9: 0.2,      # Overshooting
                    10: 0.5,     # Unknown
                    11: 0.6,     # Dust
                    12: 0.5,     # Smoke
                    15: 0.7      # N/A
                }
                cloud_norm = cloud_impact.get(cloud_type, 0.5)
                
                # Data for radar chart (repeat first value to close the polygon)
                categories = ['Solar Position', 'Temperature', 'Pressure', 
                             'Humidity', 'Wind Speed', 'Cloud Type', 'Solar Position']
                values = [solar_norm, temp_norm, pressure_norm, 
                         humidity_norm, wind_norm, cloud_norm, solar_norm]
                
                # Set up the angles for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
                
                # Plot data
                ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1E88E5')
                ax.fill(angles, values, alpha=0.25, color='#1E88E5')
                
                # Add category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories[:-1])
                
                # Make the plot look nice
                ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(['25%', '50%', '75%', '100%'])
                ax.grid(True, alpha=0.3)
                
                ax.set_title('Input Parameter Impact Profile', fontsize=12, pad=10)
                
                st.pyplot(fig)
        
        # Additional information
        st.markdown("<hr style='margin: 30px 0'>", unsafe_allow_html=True)
        st.subheader("Input Parameter Summary")
        
        # Create a more visually appealing display of the input parameters
        params_col1, params_col2, params_col3 = st.columns(3)
        
        with params_col1:
            st.markdown(f"""
            **Solar Zenith Angle**: {solar_zenith:.1f}° 
            <span style="color: gray; font-size: 0.9em;">(0° is directly overhead)</span>
            
            **Temperature**: {temperature:.1f} °C
            <span style="color: gray; font-size: 0.9em;">(Affects air density and scattering)</span>
            """, unsafe_allow_html=True)
            
        with params_col2:
            st.markdown(f"""
            **Pressure**: {pressure:.1f} mbar
            <span style="color: gray; font-size: 0.9em;">(Atmospheric pressure at measurement site)</span>
            
            **Humidity**: {humidity:.1f}%
            <span style="color: gray; font-size: 0.9em;">(Impacts atmospheric absorption)</span>
            """, unsafe_allow_html=True)
            
        with params_col3:
            st.markdown(f"""
            **Wind Speed**: {wind_speed:.1f} m/s
            <span style="color: gray; font-size: 0.9em;">(Affects convection and cooling)</span>
            
            **Cloud Type**: {cloud_type} - {cloud_types[cloud_type]}
            <span style="color: gray; font-size: 0.9em;">(Major influence on irradiance)</span>
            """, unsafe_allow_html=True)
    
    elif submit_button and not model_loaded:
        st.warning("Please wait for the model to load before making predictions.")
    
    else:
        # Welcome screen with information
        st.info("👈 Enter parameters in the sidebar and click 'Predict Solar Irradiance' to get started!")
        
        # Create an attractive explainer section
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h2 style="color: #1E88E5;">How Solar Irradiance Prediction Works</h2>
            <p style="font-size: 1.1em; margin-bottom: 20px;">
                This dashboard uses a machine learning model to predict Global Horizontal Irradiance (GHI) 
                based on atmospheric conditions. GHI is the total amount of shortwave radiation received 
                from above by a horizontal surface on the Earth.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show information about each parameter with icons
        st.markdown("<h3>Key Input Parameters</h3>", unsafe_allow_html=True)
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.markdown("""
            **☀️ Solar Zenith Angle**  
            The angle between the sun and the vertical. 0° means the sun is directly overhead, 
            90° means the sun is at the horizon. Lower angles generally result in higher irradiance.
            
            **🌡️ Temperature**  
            Air temperature affects the density of the atmosphere and how much radiation is scattered.
            
            **🌀 Pressure**  
            Atmospheric pressure affects the thickness of the atmosphere that sunlight must travel through.
            """, unsafe_allow_html=True)
            
        with param_col2:
            st.markdown("""
            **💧 Humidity**  
            Water vapor in the air absorbs and scatters solar radiation. Higher humidity generally 
            results in lower irradiance.
            
            **💨 Wind Speed**  
            Wind affects heat distribution and can influence cloud formation and movement.
            
            **☁️ Cloud Type**  
            Perhaps the most important factor. Clouds reflect, absorb, and scatter radiation, 
            with different cloud types having dramatically different effects.
            """, unsafe_allow_html=True)
        
        # Load sample prediction history if it exists
        try:
            history = get_prediction_history()
            if history and len(history) > 0:
                st.markdown("<hr style='margin: 30px 0'>", unsafe_allow_html=True)
                st.subheader("Recent Predictions")
                
                # Convert to DataFrame
                history_df = pd.DataFrame(history[-5:]).sort_values(by="timestamp", ascending=False).reset_index(drop=True)
                
                # Add cloud type description
                if 'cloud_type' in history_df.columns:
                    history_df['cloud_type'] = history_df['cloud_type'].apply(
                        lambda x: f"{x} - {cloud_types.get(x, 'Unknown')}"
                    )
                
                # Rename columns for better display
                display_cols = {
                    'timestamp': 'Timestamp',
                    'solar_zenith': 'Solar Zenith (°)',
                    'temperature': 'Temperature (°C)',
                    'cloud_type': 'Cloud Type',
                    'predicted_ghi': 'Predicted GHI (W/m²)',
                    'confidence': 'Confidence (%)'
                }
                
                # Select and rename only the most relevant columns for the preview
                preview_df = history_df[list(display_cols.keys())].rename(columns=display_cols)
                
                st.dataframe(preview_df, use_container_width=True)
                
                # Link to full history
                st.markdown("""
                <div style="text-align: right; margin-top: 10px;">
                    <a href="./Prediction_History" target="_self" style="color: #1E88E5;">
                        View complete prediction history →
                    </a>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            # Silently pass, this is just a nice-to-have feature
            pass

# Quick Statistics tab
with tab2:
    st.subheader("Input Parameter Ranges")
    
    ranges = {
        "Solar Zenith Angle": {"min": "0°", "max": "90°", "typical": "0° to 80°"},
        "Temperature": {"min": "-50°C", "max": "60°C", "typical": "0°C to 40°C"},
        "Pressure": {"min": "800 mbar", "max": "1200 mbar", "typical": "950 to 1050 mbar"},
        "Relative Humidity": {"min": "0%", "max": "100%", "typical": "30% to 90%"},
        "Wind Speed": {"min": "0 m/s", "max": "50 m/s", "typical": "0 to 15 m/s"},
    }
    
    st.table(pd.DataFrame(ranges))
    
    st.subheader("Cloud Type Influence")
    st.write("""
    Cloud types strongly affect solar irradiance. Clear skies allow maximum irradiance, 
    while overcast conditions reduce it significantly. The cloud type dropdown represents
    different atmospheric conditions detected by satellites or ground observations.
    """)
    
    # Display cloud types in columns for better readability
    cloud_df = pd.DataFrame({
        "Code": cloud_types.keys(),
        "Description": cloud_types.values()
    })
    
    st.dataframe(cloud_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Solar Irradiance Prediction Dashboard | Created with Streamlit")
