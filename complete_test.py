import streamlit as st

st.title("🔍 Complete App Test")

st.write("Testing all imports and dependencies...")

try:
    # Test basic imports
    import pandas as pd
    import numpy as np
    st.write("✅ Basic libraries: OK")
    
    # Test plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    st.write("✅ Plotly: OK")
    
    # Test sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    st.write("✅ Scikit-learn: OK")
    
    # Test other dependencies
    import joblib
    import io
    import os
    from datetime import datetime
    st.write("✅ Other dependencies: OK")
    
    # Test tab creation
    tab1, tab2 = st.tabs(["Test Tab 1", "Test Tab 2"])
    
    with tab1:
        st.write("This is tab 1")
        
        # Test file uploader
        uploaded_file = st.file_uploader("Test file upload", type=['csv'])
        if uploaded_file:
            st.write("File uploaded successfully!")
        
        # Test button
        if st.button("Test Button"):
            st.success("Button clicked!")
    
    with tab2:
        st.write("This is tab 2")
        
        # Test data display
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        st.dataframe(test_df)
        
        # Test metric display
        st.metric("Test Metric", "100%")
    
    st.success("🎉 All tests passed! App should be working correctly.")
    
except Exception as e:
    st.error(f"❌ Error found: {e}")
    import traceback
    st.code(traceback.format_exc())
