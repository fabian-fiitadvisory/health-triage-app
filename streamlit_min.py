import streamlit as st, sys, platform, numpy, pandas
st.title("✅ Streamlit Cloud is running")
st.write({
    "python": sys.version,
    "platform": platform.platform(),
    "streamlit": st.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
})
