import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Introduction
st.title('Ionising Radiation Experiment Analysis')
st.markdown("""
This app guides you through the process of analyzing data from an Ionising Radiation experiment using linear regression. 
We'll fit a linear model to the data to verify the inverse square law, visualize the fit, and discuss the statistical significance of our results.
""")

# Data Upload
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not df.empty:
        st.write("First few rows of your dataset:")
        st.dataframe(df.head())

        # Column Selection
        distance_col = st.selectbox('Select the column for distance (in meters):', df.columns)
        count_col = st.selectbox('Select the column for counts:', df.columns, index=1)

        # Data Preparation
        df['Count Rate'] = df[count_col] / 60  # Convert to count rate
        df['Inverse Square Distance'] = 1 / df[distance_col]**2

        # Linear Regression
        regression_result = linregress(df['Inverse Square Distance'], df['Count Rate'])
        
        # Visualization
        fig, ax = plt.subplots()
        ax.plot(df['Inverse Square Distance'], df['Count Rate'], 'o', label='Original data', markersize=10)
        ax.plot(df['Inverse Square Distance'], regression_result.intercept + regression_result.slope * df['Inverse Square Distance'], 'r', label='Fitted line')
        ax.set_xlabel('Inverse Square of Distance (1/m^2)')
        ax.set_ylabel('Count Rate (counts/s)')
        ax.legend()
        st.pyplot(fig)

        # Display Regression Results and explanations
        st.subheader('Regression Analysis')
        st.write(f"Slope (proportional to intensity): {regression_result.slope:.4f}")
        st.markdown("""
        **Slope (proportional to intensity):** Represents the change in count rate per unit change in the inverse square of distance. 
        A higher slope indicates a stronger relationship between the distance and count rate.
        """)

        st.write(f"Intercept: {regression_result.intercept:.4f}")
        st.markdown("""
        **Intercept:** The expected count rate when the inverse square distance is zero. Theoretically, it's an extrapolation since the inverse square distance cannot be zero.
        """)

        st.write(f"R-squared value: {regression_result.rvalue**2:.4f}")
        st.markdown("""
        **R-squared value:** Indicates the proportion of the variance in the count rate that is predictable from the inverse square distance. 
        Values closer to 1 suggest a stronger relationship.
        """)

        st.write(f"p-value of the regression: {regression_result.pvalue:.4f}")
        st.markdown("""
        **p-value of the regression:** Assesses the significance of the slope. A p-value below 0.05 typically indicates strong evidence against the null hypothesis, suggesting a significant relationship.
        """)

        st.write(f"Standard error of the estimate: {regression_result.stderr:.4f}")
        st.markdown("""
        **Standard error of the estimate:** Measures the accuracy of the regression model's predictions. Lower values indicate a model that more accurately fits the data.
        """)

        # Conclusion based on R-squared value
        if regression_result.rvalue**2 > 0.9:
            st.markdown("### The data strongly supports the inverse square law.")
        else:
            st.markdown("### The data does not strongly support the inverse square law.")
    else:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
else:
    st.info('Awaiting CSV file to be uploaded for analysis.')

# You can continue to add additional sections to provide more insights or discuss related topics.
