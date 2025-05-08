import streamlit as st
import pandas as pd

st.set_page_config(layout="centered",
                   page_title="Financial Planning Tool",
                   initial_sidebar_state="collapsed")

st.title("💼 Start your investing journey")
st.markdown("Our quiz will help you find a personalized path that fits your financial goals and risk comfort.")

if "show_tabs" not in st.session_state:
    if st.button("🚀 Start the quiz", use_container_width=True):
        st.session_state.show_tabs = True
        st.rerun()

st.markdown("---") 

# Show tabs only after Start is clicked
if st.session_state.get("show_tabs"):
    from pages.risk_assessment_app import main as run_risk_assessment
    from pages.stock_screener import main as run_stock_screener

    tab1, tab2 = st.tabs(["Risk Assessment", "Stock Screener"])

    with tab1:
        st.header("Investment Risk Assessment")
        run_risk_assessment()

    with tab2:
        st.header("Stock Screening and Analysis")
        run_stock_screener()