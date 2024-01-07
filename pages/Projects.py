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
    # source: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
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


@st.cache_resource
def load_fruit_detector_model():
    # source: https://huggingface.co/jazzmacedo/fruits-and-vegetables-detector-36?
    model = pipeline("image-classification", model="jazzmacedo/fruits-and-vegetables-detector-36")
    return model

def fruit_detector():
    st.write("## Fruit Fiasco Finder")
    st.write("Hey there! I'm the Fruit Fiasco Finder. Pop in a photo and I'll sniff out the fruits in the bunch!")

    uploaded_file = st.file_uploader("Drop a fruit image here and watch me go bananas!", type=["jpg", "png", "jpeg"], key="fruit_detector_uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Let's take a peek at this fruity puzzle...", use_column_width=True)

        if st.button("Identify the Fruit"):
            with st.spinner("Scrutinizing the scrumptious..."):
                model = load_fruit_detector_model()
                results = model(image)
                if results[0]['score'] >= 0.999:
                    st.success(f"Confidently, that's a {results[0]['label']}!")
                else:
                    descriptions = [f"{res['label']} ({res['score'] * 100:.2f}%)" for res in results[:5]]
                    st.success(f"I see: {', '.join(descriptions)}")


def pdf_merger():
    st.write("## PDF Merger")
    st.write("Hi there! I'm the PDF Merger. Upload some PDFs and I'll combine them into a single file for you!")

    uploaded_files = st.file_uploader("Drop some PDFs here and watch me go!", type=["pdf"], accept_multiple_files=True, key="pdf_merger_uploader")
    if uploaded_files is not None:
        ordered_files = st.multiselect('Order the PDFs as desired:', uploaded_files, default=uploaded_files, format_func=lambda x: x.name)
        if st.button('Merge PDFs'):
            with st.spinner("Merging..."):
                from PyPDF2 import PdfMerger
                from io import BytesIO
                merger = PdfMerger()
                for pdf in ordered_files:
                    merger.append(pdf)
                output = BytesIO()
                merger.write(output)
                merged_pdf = output.getvalue()
                st.success("Done! Click below to download the merged PDF.")
                st.download_button(label="Download Merged PDF", data=merged_pdf, file_name="merged.pdf", mime='application/pdf')


       

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

    # Use Streamlit expanders for each project
    with st.expander("Fruits Identifier Project", expanded=False):
        fruit_detector()  # The model will load when this expander is opened

    # Use Streamlit expanders for each project
    with st.expander("PDF Merger App", expanded=False):
        pdf_merger()  # The model will load when this expander is opened

    # Add more expanders for each additional project you want to include

    # Add the footer
    render_footer()

render_projects()