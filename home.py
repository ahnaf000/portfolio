import streamlit as st
#from pages import About, LearnWithMe, Projects
from footer import render_footer
st.set_page_config(layout="wide")
def main():
    
    st.title("Welcome to My Portfolio!")
    st.image("imgs/Ahnaf Professional Headshot.jpg", width=300)
    st.write("Here's a brief overview of my skills and experiences.")
    render_footer()

# The app starts here
if __name__ == "__main__":
    main()
    