import streamlit as st
import pandas as pd
import numpy as np

st.title("🛡️ CyberShield AI - Fraud Detection")
st.write("Welcome to CyberShield AI!")

# Simple test to verify dependencies work
st.write("Testing dependencies...")
st.success("✅ Streamlit: Working")

try:
    import pandas as pd
    st.success("✅ Pandas: Working")
except:
    st.error("❌ Pandas: Failed")

try:
    import numpy as np
    st.success("✅ NumPy: Working")
except:
    st.error("❌ NumPy: Failed")

try:
    import sklearn
    st.success("✅ Scikit-learn: Working")
except:
    st.error("❌ Scikit-learn: Failed")

st.info("If all dependencies are working, the main app should deploy successfully!")
