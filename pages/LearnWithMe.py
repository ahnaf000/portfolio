import streamlit as st
from footer import render_footer
# st.set_page_config(layout="wide")

# Function to display the Recommended Readings & Learning page
def render_readings():
    st.title("Recommended Readings & Learning")
    st.write("Here's a list of resources I recommend for learning new skills and improving existing ones.")
    # You can list the resources here
    render_footer()

render_readings()