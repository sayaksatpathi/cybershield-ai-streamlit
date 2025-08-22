import streamlit as st

st.set_page_config(
    page_title="🛡️ CyberShield AI Test",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CyberShield AI - Test Page")
st.success("✅ App is working! This is a test page.")

st.write("If you can see this, the basic setup is working!")

st.info("This is a simplified test version. The full app should load similarly.")

# Test imports
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import plotly
    st.success("✅ All required packages imported successfully!")
    
    st.write("Package versions:")
    st.write(f"- Pandas: {pd.__version__}")
    st.write(f"- NumPy: {np.__version__}")
    st.write(f"- Scikit-learn: {sklearn.__version__}")
    st.write(f"- Plotly: {plotly.__version__}")
    
except Exception as e:
    st.error(f"❌ Import error: {e}")

st.balloons()
