import streamlit as st
import pandas as pd
import numpy as np

st.title("🛡️ CyberShield AI - Debug Test")

st.write("Testing basic functionality...")

# Test basic operations
try:
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    st.write("✅ Pandas DataFrame creation: OK")
    st.dataframe(df)
    
    # Test numpy
    arr = np.array([1, 2, 3])
    st.write("✅ NumPy array creation: OK")
    
    # Test file upload
    uploaded_file = st.file_uploader("Test upload", type=['csv'])
    if uploaded_file:
        st.write("✅ File upload working")
    
    st.write("✅ Basic Streamlit components working!")
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())
