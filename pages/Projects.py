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

def pet_detector():
    st.write("## Furry Friend Finder")
    st.write("""
    🐾 Welcome to the Furry Friend Finder! 🐾
    Upload an image and I'll tell you if there's a cat or a dog. 
    Remember, I'm only trained to spot cats and dogs, so if you upload a picture of anything else, I might get a little confused!
    """)

    uploaded_file = st.file_uploader("Choose an image of a cat or dog...", type=["jpg", "png", "jpeg"], key="pet_detector_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        #image = image.resize((150, 150))
        st.image(image, caption='Your furry friend!', use_column_width=True)
        
        if st.button('Detect Furry Friend'):
            model = load_model()
            results = model(image)
            best_result = max(results, key=lambda x: x['score'])
            predicted_label = best_result['label']
            confidence = best_result['score'] * 100
            st.success(f"Looks like a {predicted_label}! I'm {confidence:.2f}% sure.")

@st.cache_resource
def load_image_captioning_model():

    image_to_text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return image_to_text_model
    

def image_captioning():
    st.write("## Snapshot Storyteller")
    st.write("Hello there! I'm your Snapshot Storyteller AI. Give me a photo, and I'll tell you what I see!")

    uploaded_file = st.file_uploader("Choose an image for me to describe", type=["jpg", "png", "jpeg"], key="image_captioning_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hmm, analyzing this one...", use_column_width=True)

        if st.button("Reveal the Story"):
            with st.spinner("Give me a moment to think..."):
                model = load_image_captioning_model()
                result = model(image)
                caption = result[0]['generated_text'] if result else "Oh dear, I'm not quite sure what to say about this one."
                st.success(f"Here's what I think: \"{caption}\"")



# Function to display the Projects page
def render_projects():
    st.title("Projects")
    st.write("Here are some of the projects I've worked on.")

    # Use Streamlit expanders for each project
    with st.expander("Pet Detector Project", expanded=False):
        pet_detector()  # The model will load when this expander is opened

    # Use Streamlit expanders for each project
    with st.expander("Image Captioning Project", expanded=False):
        image_captioning()  # The model will load when this expander is opened

    # Add more expanders for each additional project you want to include

    # Add the footer
    render_footer()

render_projects()