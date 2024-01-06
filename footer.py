import streamlit as st
from datetime import datetime
def render_footer():
    st.markdown("---")  # This adds a simple horizontal line to separate the footer
    
    # Use a container to hold the footer content
    with st.container():
        # Create a row with social links
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ahnaf-kabir-2590921b9/)
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            [![GitHub](https://img.shields.io/badge/GitHub-black?style=flat-square&logo=github)](https://github.com/ahnaf000)
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/abrark__/)
            """, unsafe_allow_html=True)

    # Contact information and copyright notice, smaller and closer together
    st.markdown('#### Get in Touch')
    st.caption('If you want to reach out, please feel free to send a message on LinkedIn.')
    st.caption(f'© {datetime.now().year} Your Name or Company Name. All Rights Reserved.')

    # Centering the container with the social media links and making the footer text smaller
    st.markdown("""
        <style>
            .stContainer { display: flex; justify-content: center; }
            .stMarkdown { text-align: center; }
            .css-10trblm { font-size: 0.8rem; }
        </style>
        """, unsafe_allow_html=True)

render_footer()
