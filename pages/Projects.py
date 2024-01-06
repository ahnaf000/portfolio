import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="cat_detector_uploader")
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

@st.cache_resource
def load_image_captioning_model():
    # source: https://huggingface.co/Salesforce/blip-image-captioning-large
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    # return processor, model

    image_to_text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return image_to_text_model
    

def image_captioning():
    st.write("## Image Captioning Project")
    st.write("Upload an image and the model will generate a caption for it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="image_captioning_uploader")
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Button to generate caption for the image
        if st.button('Generate Caption'):
            model = load_image_captioning_model()
            result = model(image)


            st.write(result[0]['generated_text'])
            # # Load the model only after clicking the button
            # processor, model = load_image_captioning_model()
            # # Process the image and generate the caption
            # inputs = processor(images=image, return_tensors="pt")
            # outputs = model.generate(**inputs)
            # caption = processor.decode(outputs[0], skip_special_tokens=True)
            # st.write("Caption:", caption)



# Function to display the Projects page
def render_projects():
    st.title("Projects")
    st.write("Here are some of the projects I've worked on.")

    # Use Streamlit expanders for each project
    with st.expander("Cat Detector Project", expanded=False):
        cat_detector()  # The model will load when this expander is opened

    # Use Streamlit expanders for each project
    with st.expander("Image Captioning Project", expanded=False):
        image_captioning()  # The model will load when this expander is opened

    # Add more expanders for each additional project you want to include

    # Add the footer
    render_footer()

render_projects()