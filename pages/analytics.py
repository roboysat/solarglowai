import streamlit as st

# Page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Analytics | Solar Irradiance Predictor",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import calendar

# Add the root directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_prediction_history

# Page title
st.title("📊 Solar Irradiance Analytics")
st.markdown("Analyze patterns and trends in solar irradiance predictions.")

# Load prediction history
history = get_prediction_history()

# Check if history exists
if not history:
    st.info("No prediction history available yet. Make some predictions to see analytics here!")
else:
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(history)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add derived time fields
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['date'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Seasonal Patterns", "Parameter Influence", "Prediction Trends"])
    
    # Seasonal Patterns
    with tab1:
        st.header("Seasonal and Temporal Patterns")
        st.markdown("Analyze how solar irradiance varies across different time periods.")
        
        # Only proceed if we have enough data
        if len(df) < 3:
            st.warning("Not enough prediction data for meaningful seasonal analysis. Make more predictions to see patterns.")
        else:
            # Time period selector
            time_period = st.radio(
                "Select Time Period",
                options=["Hourly", "Daily", "Monthly"],
                horizontal=True
            )
            
            if time_period == "Hourly":
                # Hourly patterns
                hourly_df = df.groupby('hour')['predicted_ghi'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(
                    hourly_df['hour'],
                    hourly_df['predicted_ghi'],
                    marker='o',
                    linestyle='-',
                    color='#1E88E5'
                )
                
                # Add shaded range if we have multiple predictions per hour
                if len(df) > len(hourly_df):
                    hourly_min = df.groupby('hour')['predicted_ghi'].min().reset_index()
                    hourly_max = df.groupby('hour')['predicted_ghi'].max().reset_index()
                    
                    ax.fill_between(
                        hourly_df['hour'],
                        hourly_min['predicted_ghi'],
                        hourly_max['predicted_ghi'],
                        alpha=0.2,
                        color='#1E88E5'
                    )
                
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Average Predicted GHI (W/m²)')
                ax.set_title('Average Solar Irradiance by Hour of Day')
                ax.set_xticks(range(0, 24))
                ax.grid(True, alpha=0.3)
                
                # Add vertical lines for sunrise and sunset (approximate)
                ax.axvspan(0, 6, alpha=0.2, color='gray', label='Night')
                ax.axvspan(18, 24, alpha=0.2, color='gray')
                
                st.pyplot(fig)
                
                # Additional context
                st.markdown("""
                **Hourly Analysis Insights:**
                
                - Solar irradiance typically peaks around mid-day when the sun is highest in the sky.
                - The irradiance curve follows a bell shape pattern over the day.
                - Early morning and late afternoon have lower irradiance due to higher solar zenith angles.
                """)
                
            elif time_period == "Daily":
                # Daily patterns by day of week
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily_df = df.groupby('day_of_week')['predicted_ghi'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.bar(
                    daily_df['day_of_week'],
                    daily_df['predicted_ghi'],
                    alpha=0.7,
                    color='#1E88E5'
                )
                
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Average Predicted GHI (W/m²)')
                ax.set_title('Average Solar Irradiance by Day of Week')
                ax.set_xticks(range(7))
                ax.set_xticklabels(day_names)
                ax.grid(True, axis='y', alpha=0.3)
                
                st.pyplot(fig)
                
                # Additional context
                st.markdown("""
                **Daily Analysis Insights:**
                
                - Day-to-day variations in solar irradiance are primarily due to changing weather conditions.
                - Any patterns observed across days of the week are coincidental rather than causal.
                - For meaningful daily patterns, long-term data collection is necessary.
                """)
                
            else:  # Monthly
                # Monthly patterns
                month_names = list(calendar.month_name)[1:]
                monthly_data = []
                
                # Get unique months in the dataset
                unique_months = df['month'].unique()
                
                if len(unique_months) > 1:
                    monthly_df = df.groupby('month')['predicted_ghi'].mean().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.bar(
                        monthly_df['month'],
                        monthly_df['predicted_ghi'],
                        alpha=0.7,
                        color='#1E88E5'
                    )
                    
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Predicted GHI (W/m²)')
                    ax.set_title('Average Solar Irradiance by Month')
                    ax.set_xticks(range(1, 13))
                    ax.set_xticklabels([month_names[i-1] for i in range(1, 13)], rotation=45)
                    ax.grid(True, axis='y', alpha=0.3)
                    
                    # Highlight only the months we have data for
                    for month in range(1, 13):
                        if month not in unique_months:
                            ax.get_xticklabels()[month-1].set_color('gray')
                            ax.get_xticklabels()[month-1].set_alpha(0.5)
                    
                    st.pyplot(fig)
                else:
                    st.info("Data only available for one month. Collect data across multiple months to see seasonal patterns.")
                
                # Additional context about seasonal patterns
                st.markdown("""
                **Monthly Analysis Insights:**
                
                - Solar irradiance typically peaks in summer months when days are longer and the sun is higher in the sky.
                - Winter months generally have lower irradiance due to shorter days and lower sun angles.
                - Local climate factors like monsoon seasons can create regional variations in this pattern.
                """)
            
            # Cloud type impact by season
            if 'cloud_type' in df.columns and len(df['cloud_type'].unique()) > 1:
                st.subheader("Cloud Type Impact by Season")
                
                # Define seasons (simplified to be based on month)
                def get_season(month):
                    if month in [12, 1, 2]:
                        return "Winter"
                    elif month in [3, 4, 5]:
                        return "Spring"
                    elif month in [6, 7, 8]:
                        return "Summer"
                    else:
                        return "Fall"
                
                df['season'] = df['month'].apply(get_season)
                
                # Cloud type mapping
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
                
                # Get unique seasons in the dataset
                unique_seasons = df['season'].unique()
                
                if len(unique_seasons) > 1:
                    # Pivot table for season vs cloud type
                    cloud_season_pivot = df.pivot_table(
                        values='predicted_ghi',
                        index='cloud_type',
                        columns='season',
                        aggfunc='mean'
                    ).reset_index()
                    
                    # Replace cloud type codes with descriptions
                    cloud_season_pivot['cloud_type'] = cloud_season_pivot['cloud_type'].map(
                        lambda x: f"{x} - {cloud_types.get(x, 'Unknown')}"
                    )
                    
                    # Display as a table
                    st.dataframe(cloud_season_pivot.set_index('cloud_type'), use_container_width=True)
                    
                    # Create a grouped bar chart
                    cloud_types_in_data = sorted(df['cloud_type'].unique())
                    
                    # Get the data in the right format for plotting
                    plot_data = []
                    for season in unique_seasons:
                        season_data = []
                        for ct in cloud_types_in_data:
                            subset = df[(df['season'] == season) & (df['cloud_type'] == ct)]
                            if len(subset) > 0:
                                season_data.append(subset['predicted_ghi'].mean())
                            else:
                                season_data.append(0)
                        plot_data.append(season_data)
                    
                    # Only proceed if we have enough data
                    if any(any(data > 0 for data in season_data) for season_data in plot_data):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        x = np.arange(len(cloud_types_in_data))
                        width = 0.8 / len(unique_seasons)
                        
                        for i, season in enumerate(unique_seasons):
                            ax.bar(
                                x + i * width - 0.4 + width/2,
                                plot_data[i],
                                width,
                                label=season,
                                alpha=0.7
                            )
                        
                        # Set labels with cloud type descriptions
                        labels = [f"{ct} - {cloud_types.get(ct, 'Unknown')}" for ct in cloud_types_in_data]
                        
                        ax.set_xlabel('Cloud Type')
                        ax.set_ylabel('Average Predicted GHI (W/m²)')
                        ax.set_title('Impact of Cloud Type on Solar Irradiance by Season')
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.legend()
                        ax.grid(True, axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("Data only available for one season. Collect data across multiple seasons to see seasonal cloud type impact.")
    
    # Parameter Influence Analysis
    with tab2:
        st.header("Parameter Influence Analysis")
        st.markdown("Analyze how different atmospheric parameters affect solar irradiance predictions.")
        
        # Select parameters to analyze
        parameters = [
            ('solar_zenith', 'Solar Zenith Angle (°)'),
            ('temperature', 'Temperature (°C)'),
            ('pressure', 'Pressure (mbar)'),
            ('humidity', 'Humidity (%)'),
            ('wind_speed', 'Wind Speed (m/s)'),
            ('cloud_type', 'Cloud Type')
        ]
        
        param1, param2 = st.columns(2)
        
        with param1:
            selected_param1 = st.selectbox(
                "Select First Parameter",
                options=[p[0] for p in parameters],
                format_func=lambda x: dict(parameters)[x],
                index=0
            )
        
        with param2:
            # Filter out the first selected parameter
            remaining_params = [p for p in parameters if p[0] != selected_param1]
            selected_param2 = st.selectbox(
                "Select Second Parameter",
                options=[p[0] for p in remaining_params],
                format_func=lambda x: dict(parameters)[x],
                index=0
            )
        
        # Plot the relationship
        st.subheader(f"Relationship: {dict(parameters)[selected_param1]} vs. {dict(parameters)[selected_param2]}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Different handling based on parameter types
        if selected_param1 == 'cloud_type' and selected_param2 == 'cloud_type':
            st.warning("Please select different parameters for comparison.")
        elif selected_param1 == 'cloud_type':
            # Box plot for cloud type vs numeric parameter
            cloud_types_in_data = sorted(df['cloud_type'].unique())
            
            # Prepare data for boxplot
            boxplot_data = [df[df['cloud_type'] == ct][selected_param2] for ct in cloud_types_in_data]
            
            # Create box plot
            ax.boxplot(boxplot_data)
            
            # Set x-ticks with cloud type descriptions
            cloud_type_dict = {
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
            
            labels = [f"{ct} - {cloud_type_dict.get(ct, 'Unknown')}" for ct in cloud_types_in_data]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_xlabel(dict(parameters)[selected_param1])
            ax.set_ylabel(dict(parameters)[selected_param2])
            
        elif selected_param2 == 'cloud_type':
            # Box plot for numeric parameter vs cloud type
            cloud_types_in_data = sorted(df['cloud_type'].unique())
            
            # Prepare data for boxplot
            boxplot_data = [df[df['cloud_type'] == ct][selected_param1] for ct in cloud_types_in_data]
            
            # Create box plot
            ax.boxplot(boxplot_data)
            
            # Set x-ticks with cloud type descriptions
            cloud_type_dict = {
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
            
            labels = [f"{ct} - {cloud_type_dict.get(ct, 'Unknown')}" for ct in cloud_types_in_data]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_xlabel(dict(parameters)[selected_param2])
            ax.set_ylabel(dict(parameters)[selected_param1])
            
        else:
            # Scatter plot for two numeric parameters
            scatter = ax.scatter(
                df[selected_param1],
                df[selected_param2],
                c=df['predicted_ghi'],
                cmap='viridis',
                alpha=0.7
            )
            
            ax.set_xlabel(dict(parameters)[selected_param1])
            ax.set_ylabel(dict(parameters)[selected_param2])
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Predicted GHI (W/m²)')
        
        ax.set_title(f'Relationship between {dict(parameters)[selected_param1]} and {dict(parameters)[selected_param2]}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Parameter correlation heatmap
        st.subheader("Parameter Correlation Analysis")
        
        # Only use numeric columns for correlation
        numeric_columns = ['solar_zenith', 'temperature', 'pressure', 'humidity', 'wind_speed', 'predicted_ghi']
        numeric_df = df[numeric_columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Add labels
        parameter_labels = {
            'solar_zenith': 'Solar Zenith',
            'temperature': 'Temperature',
            'pressure': 'Pressure',
            'humidity': 'Humidity',
            'wind_speed': 'Wind Speed',
            'predicted_ghi': 'Predicted GHI'
        }
        
        ax.set_xticks(np.arange(len(numeric_columns)))
        ax.set_yticks(np.arange(len(numeric_columns)))
        ax.set_xticklabels([parameter_labels[col] for col in numeric_columns], rotation=45, ha='right')
        ax.set_yticklabels([parameter_labels[col] for col in numeric_columns])
        
        # Add correlation values in the cells
        for i in range(len(numeric_columns)):
            for j in range(len(numeric_columns)):
                text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                              ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
        
        ax.set_title('Correlation Matrix of Input Parameters and Predicted GHI')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display interpretation
        st.markdown("""
        **Correlation Interpretation Guide:**
        
        - Values close to 1 indicate a strong positive correlation (as one parameter increases, the other also increases).
        - Values close to -1 indicate a strong negative correlation (as one parameter increases, the other decreases).
        - Values close to 0 indicate little to no correlation.
        
        **Key Relationships to Look For:**
        
        - Solar Zenith Angle vs. GHI: Typically a strong negative correlation, as higher zenith angles (sun lower in sky) result in lower irradiance.
        - Temperature vs. GHI: Often positively correlated, as more solar radiation tends to increase temperature.
        - Cloud Type vs. GHI: Different cloud types significantly affect solar irradiance, with clear skies allowing maximum radiation.
        """)
    
    # Prediction Trends
    with tab3:
        st.header("Prediction Trends and Analysis")
        
        # Time range selector
        st.subheader("Select Time Range")
        
        # Get the min and max dates from the dataframe
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        date_range = st.date_input(
            "Date Range",
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
        
        # Resample data by day for trend analysis
        daily_df = filtered_df.groupby('date')['predicted_ghi'].agg(['mean', 'min', 'max', 'std']).reset_index()
        
        # Plot the trend
        st.subheader("Solar Irradiance Prediction Trend")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean GHI with min-max range
        ax.plot(
            daily_df['date'],
            daily_df['mean'],
            marker='o',
            linestyle='-',
            color='#1E88E5',
            label='Mean Predicted GHI'
        )
        
        # Add min-max range if we have more than one prediction per day
        if len(filtered_df) > len(daily_df):
            ax.fill_between(
                daily_df['date'],
                daily_df['min'],
                daily_df['max'],
                alpha=0.2,
                color='#1E88E5',
                label='Min-Max Range'
            )
        
        # Add trend line
        if len(daily_df) > 1:
            # Convert dates to numeric for regression
            date_nums = np.array([(d - min_date).days for d in daily_df['date']])
            
            # Simple linear regression
            z = np.polyfit(date_nums, daily_df['mean'], 1)
            p = np.poly1d(z)
            
            # Add trend line to plot
            ax.plot(
                daily_df['date'],
                p(date_nums),
                linestyle='--',
                color='red',
                label=f'Trend Line (Slope: {z[0]:.4f})'
            )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted GHI (W/m²)')
        ax.set_title('Solar Irradiance Prediction Trend Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Calculate metrics
        mean_ghi = filtered_df['predicted_ghi'].mean()
        median_ghi = filtered_df['predicted_ghi'].median()
        std_ghi = filtered_df['predicted_ghi'].std()
        max_ghi = filtered_df['predicted_ghi'].max()
        min_ghi = filtered_df['predicted_ghi'].min()
        
        # Display metrics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean GHI", f"{mean_ghi:.2f} W/m²")
            st.metric("Min GHI", f"{min_ghi:.2f} W/m²")
        
        with col2:
            st.metric("Median GHI", f"{median_ghi:.2f} W/m²")
            st.metric("Max GHI", f"{max_ghi:.2f} W/m²")
        
        with col3:
            st.metric("Standard Deviation", f"{std_ghi:.2f} W/m²")
            st.metric("Range", f"{max_ghi - min_ghi:.2f} W/m²")
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(
            filtered_df['predicted_ghi'],
            bins=20,
            alpha=0.7,
            color='#1E88E5'
        )
        
        # Add vertical lines for mean and median
        ax.axvline(x=mean_ghi, color='red', linestyle='--', label=f'Mean: {mean_ghi:.2f}')
        ax.axvline(x=median_ghi, color='green', linestyle='-.', label=f'Median: {median_ghi:.2f}')
        
        ax.set_xlabel('Predicted GHI (W/m²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predicted Solar Irradiance Values')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # If we have actual GHI values, show prediction accuracy over time
        if 'actual_ghi' in filtered_df.columns:
            accuracy_df = filtered_df.dropna(subset=['actual_ghi'])
            
            if not accuracy_df.empty:
                st.subheader("Prediction Accuracy Over Time")
                
                # Calculate error and percentage error
                accuracy_df['error'] = accuracy_df['predicted_ghi'] - accuracy_df['actual_ghi']
                accuracy_df['abs_error'] = np.abs(accuracy_df['error'])
                accuracy_df['pct_error'] = (accuracy_df['abs_error'] / accuracy_df['actual_ghi']) * 100
                
                # Resample by day
                daily_error_df = accuracy_df.groupby('date')['pct_error'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(
                    daily_error_df['date'],
                    daily_error_df['pct_error'],
                    marker='o',
                    linestyle='-',
                    color='#1E88E5'
                )
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Mean Percentage Error (%)')
                ax.set_title('Prediction Accuracy Over Time')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis date labels
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Solar Irradiance Prediction Dashboard | Analytics | Created with Streamlit")
