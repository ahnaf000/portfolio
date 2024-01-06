import streamlit as st
from footer import render_footer
# st.set_page_config(layout="wide")

# Function to display the About page
def render_about():
    st.title("About Me")
    st.write("Here's something more about me.")
    # You can add more personal details here
    render_footer()

render_about()