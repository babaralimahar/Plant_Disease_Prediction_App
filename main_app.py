# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model # Modernized import

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for aesthetic tweaks ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model with Caching ---
# Caching prevents the model from reloading every time the user interacts with the app
@st.cache_resource
def load_disease_model():
    # Use a relative path for cloud deployment
    return load_model('plant_disease.h5')

try:
    model = load_disease_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Name of Classes
CLASS_NAMES = ['Corn - Common Rust', 'Potato - Early Blight', 'Tomato - Bacterial Spot']

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100) # Placeholder logo
    st.title("About the App")
    st.info(
        "This AI-powered tool helps farmers and gardeners quickly identify "
        "common diseases in Corn, Potato, and Tomato plants by analyzing leaf images."
    )
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Upload a clear image of the diseased leaf.")
    st.markdown("2. Click the 'Analyze Leaf' button.")
    st.markdown("3. Review the AI prediction and confidence score.")

# --- Main App UI ---
st.title("🌿 Plant Disease Detection AI")
st.markdown("Upload a high-quality image of a plant leaf to detect potential diseases.")

# Uploading the image
plant_image = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if plant_image is not None:
    # Create two columns for a side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Convert BGR to RGB for correct color rendering in Streamlit
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Displaying the image nicely with rounded corners (via use_column_width)
        st.image(rgb_image, use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        submit = st.button('🔍 Analyze Leaf')
        
        # On predict button click
        if submit:
            with st.spinner("Analyzing image patterns..."):
                # Resizing the image to match model input shape
                resized_image = cv2.resize(opencv_image, (256, 256))
                
                # Convert image to 4 Dimension (Batch size, Height, Width, Channels)
                input_image = np.expand_dims(resized_image, axis=0)
                
                # Make Prediction
                Y_pred = model.predict(input_image)
                
                # Get the class and confidence score
                predicted_class_index = np.argmax(Y_pred)
                result = CLASS_NAMES[predicted_class_index]
                confidence = np.max(Y_pred) * 100 # Assuming output is softmax probabilities
                
                # Split the result string for a cleaner display
                plant_type, disease_name = result.split(' - ')
                
                # Display Results in a premium way
                st.success("Analysis Complete!")
                
                st.markdown(f"### **Plant Type:** {plant_type}")
                st.markdown(f"### **Condition:** {disease_name}")
                
                # Show confidence as a progress bar
                st.write(f"**AI Confidence:** {confidence:.2f}%")
                st.progress(int(confidence))
                
                # Provide contextual advice (Optional, adds a lot of value)
                with st.expander("View Recommended Actions"):
                    st.write(f"Based on the detection of **{disease_name}**, consider consulting local agricultural guidelines for appropriate fungicidal or bacterial treatments suitable for **{plant_type}**.")
