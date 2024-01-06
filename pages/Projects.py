import streamlit as st
from transformers import pipeline
from PIL import Image
from footer import render_footer
# st.set_page_config(layout="wide")

# Function to load the model (this is just a placeholder, adjust according to your model)
@st.cache_resource
def load_model():
    # Replace 'image-classification' with your specific model from Hugging Face
    models = [
        "akahana/vit-base-cats-vs-dogs",
        "ismgar01/vit-base-cats-vs-dogs",
        "nateraw/vit-base-cats-vs-dogs"
    ]
    model = pipeline('image-classification', model=models[0])
    return model

def cat_detector():
    st.write("## Cat Detector Project")
    st.write("Upload an image of a cat and the model will classify if it's a cat or not.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file).convert('RGB')
        # Resize the image
        image = image.resize((150, 150))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Button to classify the image
        if st.button('Classify Image'):
            # Load the model only after clicking the button
            model = load_model()
            # Perform the classification directly on the PIL image
            results = model(image)
            # Display the classification result
            # Find the result with the highest score
            best_result = max(results, key=lambda x: x['score'])

            # Extract label and score
            predicted_label = best_result['label']
            confidence = best_result['score']

            # Convert the score to a percentage
            confidence_percent = confidence * 100

            # Create the output statement
            output_statement = f"It is a {predicted_label} and I am {confidence_percent:.2f}% confident."
            st.write(output_statement)

# Function to display the Projects page
def render_projects():
    st.title("Projects")
    st.write("Here are some of the projects I've worked on.")

    # Use Streamlit expanders for each project
    with st.expander("Cat Detector Project", expanded=False):
        cat_detector()  # The model will load when this expander is opened

    # You can add additional projects in a similar manner
    with st.expander("Another Machine Learning Project"):
        st.write("Description and functionality for another ML project will go here.")

    # Add more expanders for each additional project you want to include

    # Add the footer
    render_footer()

render_projects()