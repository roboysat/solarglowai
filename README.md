# SolarGlow AI ☀️

SolarGlow AI addresses the critical challenge of solar power unpredictability in regions with volatile weather. By providing accurate solar irradiance forecasts, it enables more stable grid operations and promotes the reliable deployment of renewable energy.

## 📖 Table of Contents

1. [The Problem](#the-problem)
2. [Our Solution](#our-solution)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [The Web Application](#the-web-application)
6. [Project Team](#project-team)

## 1. The Problem

The primary barrier to the widespread adoption of solar energy is its intermittency. Unpredictable weather patterns, especially cloud cover, cause significant fluctuations in solar power generation. This variability makes it difficult for grid operators to manage energy supply, often leading to grid instability and a continued reliance on fossil fuels for backup power. Accurate, real-time forecasting of solar irradiance is essential to mitigate these issues and unlock the full potential of solar energy.

## 2. Our Solution

SolarGlow AI employs a Long Short-Term Memory (LSTM) neural network, a sophisticated deep learning model ideal for time-series forecasting. Our approach is focused and precise:

- **Single Model Focus**: We utilize a single, powerful LSTM model specifically trained to predict solar irradiance.

- **Data Integration**: The model is trained on a rich dataset from NSRDB: National Solar Radiation Database.

By accurately forecasting this key metric, SolarGlow AI provides grid operators with the crucial foresight needed to manage power distribution, optimize energy storage, and ensure a stable and reliable supply of clean energy.

## 3. Key Features

- **High-Accuracy Forecasting**: Leverages a specialized LSTM model for precise short-term and medium-term solar irradiance predictions.

- **Enhanced Grid Stability**: Delivers actionable data to help operators anticipate fluctuations and maintain a balanced power supply.

- **Interactive Dashboard**: A user-friendly web interface to visualize forecasts and analyze data.

## 4. Technology Stack

- **Core Model**: Python
- **Deep Learning**: PyTorch, Scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib
- **Web App & UI**: Streamlit

## 5. The Web Application

To make our model accessible and easy to use, we developed an interactive web dashboard using Streamlit. The user interface allows for the visualization of both historical data and future solar irradiance forecasts.

The initial implementation of the Streamlit dashboard was bootstrapped with the assistance of the Replit AI Agent, which accelerated the development of the front-end components.

## 6. Project Team

This project was a collaborative effort by the members of Team Blaze from CVR College of Engineering:

- **Sathvik Yellani** - @roboysat
- **Dhanakshi Maheshwari**
- **Tejasvvi Sarvasiddi**
- **Krishna Kondi**
