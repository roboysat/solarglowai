import streamlit as st

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Prediction History | Solar Irradiance Predictor",
    page_icon="📜",
    layout="wide"
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
import os

# Add the root directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_prediction_history

# Page title
st.title("📜 Prediction History")
st.markdown("View and analyze your past solar irradiance predictions.")

# Load prediction history
history = get_prediction_history()

# Check if history exists
if not history:
    st.info("No prediction history available yet. Make some predictions to see them here!")
else:
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(history)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add a date column for filtering
    df['date'] = df['timestamp'].dt.date
    
    # Sort by timestamp, newest first
    df = df.sort_values('timestamp', ascending=False)
    
    # Display controls in sidebar
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Ensure we have a start and end date
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        filtered_df = df
    
    # Cloud type filter
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
    
    all_cloud_types = sorted(filtered_df['cloud_type'].unique())
    selected_cloud_types = st.sidebar.multiselect(
        "Filter by Cloud Type",
        options=all_cloud_types,
        format_func=lambda x: f"{x} - {cloud_types.get(x, 'Unknown')}",
        default=all_cloud_types
    )
    
    if selected_cloud_types:
        filtered_df = filtered_df[filtered_df['cloud_type'].isin(selected_cloud_types)]
    
    # Summary statistics
    st.header("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", len(filtered_df))
        
    with col2:
        mean_ghi = filtered_df['predicted_ghi'].mean()
        st.metric("Average Predicted GHI", f"{mean_ghi:.2f} W/m²")
        
    with col3:
        mean_confidence = filtered_df['confidence'].mean()
        st.metric("Average Confidence", f"{mean_confidence:.2f}%")
    
    # Predictions table
    st.header("Prediction Records")
    
    # Format the dataframe for display
    display_df = filtered_df.copy()
    
    # Rename columns for better readability
    display_df = display_df.rename(columns={
        'timestamp': 'Timestamp',
        'solar_zenith': 'Solar Zenith (°)',
        'temperature': 'Temperature (°C)',
        'pressure': 'Pressure (mbar)',
        'humidity': 'Humidity (%)',
        'wind_speed': 'Wind Speed (m/s)',
        'cloud_type': 'Cloud Type',
        'predicted_ghi': 'Predicted GHI (W/m²)',
        'confidence': 'Confidence (%)'
    })
    
    # Add cloud type description
    display_df['Cloud Type'] = display_df['Cloud Type'].apply(
        lambda x: f"{x} - {cloud_types.get(x, 'Unknown')}"
    )
    
    # If actual_ghi exists, add it to the display
    if 'actual_ghi' in display_df.columns:
        display_df = display_df.rename(columns={'actual_ghi': 'Actual GHI (W/m²)'})
    
    # Format timestamp
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Remove date column from display
    if 'date' in display_df.columns:
        display_df = display_df.drop('date', axis=1)
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Data visualizations
    st.header("Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["GHI Over Time", "Prediction Accuracy", "Parameter Distributions"])
    
    with tab1:
        # Plot GHI predictions over time
        st.subheader("Solar Irradiance Predictions Over Time")
        
        # Prepare data for plotting
        plot_df = filtered_df.sort_values('timestamp')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot predicted GHI
        ax.plot(
            plot_df['timestamp'], 
            plot_df['predicted_ghi'], 
            marker='o', 
            linestyle='-', 
            color='#1E88E5',
            label='Predicted GHI'
        )
        
        # Plot actual GHI if available
        if 'actual_ghi' in plot_df.columns:
            # Drop rows with NaN actual_ghi values
            actual_df = plot_df.dropna(subset=['actual_ghi'])
            
            if not actual_df.empty:
                ax.plot(
                    actual_df['timestamp'],
                    actual_df['actual_ghi'],
                    marker='x',
                    linestyle='--',
                    color='#FFA726',
                    label='Actual GHI'
                )
        
        # Customize plot
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('GHI (W/m²)')
        ax.set_title('Solar Irradiance Predictions Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with tab2:
        # Show prediction accuracy if actual values are available
        if 'actual_ghi' in filtered_df.columns:
            accuracy_df = filtered_df.dropna(subset=['actual_ghi'])
            
            if not accuracy_df.empty:
                st.subheader("Prediction Accuracy")
                
                # Calculate error metrics
                accuracy_df['error'] = accuracy_df['predicted_ghi'] - accuracy_df['actual_ghi']
                accuracy_df['abs_error'] = np.abs(accuracy_df['error'])
                accuracy_df['pct_error'] = (accuracy_df['abs_error'] / accuracy_df['actual_ghi']) * 100
                
                # Display error metrics
                mean_abs_error = accuracy_df['abs_error'].mean()
                mean_pct_error = accuracy_df['pct_error'].mean()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean Absolute Error", f"{mean_abs_error:.2f} W/m²")
                
                with col2:
                    st.metric("Mean Percentage Error", f"{mean_pct_error:.2f}%")
                
                # Plot predicted vs actual
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Scatter plot
                ax.scatter(
                    accuracy_df['actual_ghi'],
                    accuracy_df['predicted_ghi'],
                    color='#1E88E5',
                    alpha=0.7
                )
                
                # Add perfect prediction line
                min_val = min(accuracy_df['actual_ghi'].min(), accuracy_df['predicted_ghi'].min())
                max_val = max(accuracy_df['actual_ghi'].max(), accuracy_df['predicted_ghi'].max())
                
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                
                # Customize plot
                ax.set_xlabel('Actual GHI (W/m²)')
                ax.set_ylabel('Predicted GHI (W/m²)')
                ax.set_title('Predicted vs Actual GHI')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Error distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.hist(accuracy_df['error'], bins=20, color='#1E88E5', alpha=0.7)
                
                ax.set_xlabel('Prediction Error (W/m²)')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Prediction Errors')
                ax.grid(True, alpha=0.3)
                
                # Add vertical line at zero
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            else:
                st.info("No records with actual GHI values available for accuracy analysis.")
        else:
            st.info("Accuracy metrics are only available when predictions include actual GHI values.")
    
    with tab3:
        st.subheader("Input Parameter Distributions")
        
        # Select parameter to visualize
        parameter = st.selectbox(
            "Select Parameter",
            options=[
                'solar_zenith',
                'temperature',
                'pressure',
                'humidity',
                'wind_speed',
                'cloud_type'
            ],
            format_func=lambda x: {
                'solar_zenith': 'Solar Zenith Angle (°)',
                'temperature': 'Temperature (°C)',
                'pressure': 'Pressure (mbar)',
                'humidity': 'Humidity (%)',
                'wind_speed': 'Wind Speed (m/s)',
                'cloud_type': 'Cloud Type'
            }.get(x, x)
        )
        
        # Plot histogram of the selected parameter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For cloud type, use bar chart instead of histogram
        if parameter == 'cloud_type':
            cloud_counts = filtered_df['cloud_type'].value_counts().sort_index()
            
            # Create labels with cloud type descriptions
            labels = [f"{ct} - {cloud_types.get(ct, 'Unknown')}" for ct in cloud_counts.index]
            
            ax.bar(range(len(cloud_counts)), cloud_counts.values, alpha=0.7, color='#1E88E5')
            ax.set_xticks(range(len(cloud_counts)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax.hist(filtered_df[parameter], bins=20, alpha=0.7, color='#1E88E5')
        
        # Set labels based on parameter
        param_labels = {
            'solar_zenith': 'Solar Zenith Angle (°)',
            'temperature': 'Temperature (°C)',
            'pressure': 'Pressure (mbar)',
            'humidity': 'Humidity (%)',
            'wind_speed': 'Wind Speed (m/s)',
            'cloud_type': 'Cloud Type'
        }
        
        ax.set_xlabel(param_labels.get(parameter, parameter))
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {param_labels.get(parameter, parameter)}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show relationship between parameter and GHI
        st.subheader(f"Relationship: {param_labels.get(parameter, parameter)} vs. GHI")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For cloud type, use box plot
        if parameter == 'cloud_type':
            # Only include cloud types that are in the dataset
            available_types = sorted(filtered_df['cloud_type'].unique())
            
            # Prepare data for boxplot
            boxplot_data = [filtered_df[filtered_df['cloud_type'] == ct]['predicted_ghi'] for ct in available_types]
            
            # Create box plot
            ax.boxplot(boxplot_data)
            
            # Set x-ticks with cloud type descriptions
            labels = [f"{ct} - {cloud_types.get(ct, 'Unknown')}" for ct in available_types]
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            # Scatter plot for numeric parameters
            ax.scatter(filtered_df[parameter], filtered_df['predicted_ghi'], alpha=0.7, color='#1E88E5')
        
        ax.set_xlabel(param_labels.get(parameter, parameter))
        ax.set_ylabel('Predicted GHI (W/m²)')
        ax.set_title(f'{param_labels.get(parameter, parameter)} vs. Predicted GHI')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Export functionality
    st.header("Export Data")
    
    # Create a download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Prediction History as CSV",
        data=csv,
        file_name=f"solar_prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.caption("Solar Irradiance Prediction Dashboard | Prediction History | Created with Streamlit")
