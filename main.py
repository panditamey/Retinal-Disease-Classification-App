import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.layers import DepthwiseConv2D
import numpy as np
import tempfile
import os

# # Define a custom DepthwiseConv2D layer class
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, **kwargs):
#         # Remove the 'groups' argument if present
#         kwargs.pop('groups', None)
#         super().__init__(**kwargs)

# # Define custom objects used in the model
# custom_objects = {
#     'CustomDepthwiseConv2D': CustomDepthwiseConv2D
# }
# Load the pre-trained model
model_loaded = load_model('model.h5')

# Define the image classification function
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.reshape((1, 224, 224, 3))
    x = x / 255.0
    prediction = model_loaded.predict(x)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    predicted_class = classes[predicted_class_index]
    return predicted_class

# Streamlit app
def main():
    st.title("Retinal Disease Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Perform classification on the uploaded image
        st.write("")
        st.write("Classifying...")

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Perform classification
        predicted_class = classify_image(temp_file_path)

        # Display the result
        st.write("Prediction:", predicted_class)

        # Delete the temporary file
        os.remove(temp_file_path)

# Run the Streamlit app
if __name__ == "__main__":
    main()
