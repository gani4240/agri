import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "soil_classification_model.h5"
model = keras.models.load_model(MODEL_PATH)

# Define image parameters
img_height, img_width = 150, 150

# Manually define class indices (since train_generator is unavailable in this script)
class_indices = {
    0: "Alluvial soil",
    1: "Black soil",
    2: "Clay soil",
    3: "Red soil"
}

# Define the soil type to crop mapping
soil_to_crop = {
    "Alluvial soil": ["Wheat", "Maize", "Rice", "Sugarcane"],
    "Black soil": ["Cotton", "Jowar", "Sunflower"],
    "Clay soil": ["Rice", "Soybean", "Cabbage","Turmeric","Tomato","Cotton"],
    "Red soil": ["Rice","Millets", "Pulses", "Groundnut"]
}

# Streamlit UI
st.title("🌱 Soil Type Classification & Crop Recommendation 🌾")
st.write("Upload an image of soil, and the model will predict its type and suggest suitable crops.")

uploaded_file = st.file_uploader("📤 Upload a soil image", type=["jpg", "png", "jpeg"])

def predict_soil_and_crop(image):
    """Predicts the soil type from an image and suggests suitable crops."""
    img = image.resize((img_height, img_width))  # Resize image
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    predicted_label = class_indices[np.argmax(prediction)]
    
    recommended_crops = soil_to_crop.get(predicted_label, ["No recommendation available"])
    return predicted_label, recommended_crops

if uploaded_file is not None:
    # Convert the uploaded file into an image
    image = Image.open(uploaded_file)
    
    # Predict soil type and recommended crops
    soil_type, crops = predict_soil_and_crop(image)

    # Display image and predictions
    st.image(uploaded_file, caption=f"🧑‍🌾 Predicted Soil Type: {soil_type}", use_column_width=True)
    st.subheader(f"🌍 Soil Type: {soil_type}")
    st.write(f"🌿 Recommended Crops: {', '.join(crops)}")
