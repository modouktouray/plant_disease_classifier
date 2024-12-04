import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
 
# Load the trained model
model = load_model('model/mobilenetv2_plantdiseases.keras')
# Replace with your model path
 
# Define class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
 
# Create a function to preprocess the image
def preprocess_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img
 
# Streamlit app
st.title("Plant Disease Classification")
 
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
 
if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)
 
    # Preprocess the image
    processed_image = preprocess_image(image)
 
    # Make prediction
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[class_idx]
 
    # Display the prediction
    st.write(f"**Prediction:**{predicted_class}")