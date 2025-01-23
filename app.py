import streamlit as st
from tensorflow.keras.models import load_model
from encoder import EncoderNetwork
from decoder import DecoderNetwork
from PIL import Image
import numpy as np

# Load the steganography model
st.title("End-to-End Steganography App")
st.write("Upload a payload image and a carrier image to encode and decode messages.")

@st.cache_resource
def load_steganography_model():
    return load_model("steganography_model.h5", compile=False)

steganography_model = load_steganography_model()

# Create encoder and decoder instances
encoder = EncoderNetwork()
decoder = DecoderNetwork()

# Image Preprocessing Function
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

# File Upload Section
st.header("1. Upload Images")

payload_file = st.file_uploader("Upload Payload Image (grayscale)", type=["png", "jpg", "jpeg"])
carrier_file = st.file_uploader("Upload Carrier Image (RGB)", type=["png", "jpg", "jpeg"])

if payload_file and carrier_file:
    payload_image = Image.open(payload_file).convert("L")  # Convert to grayscale
    carrier_image = Image.open(carrier_file).convert("RGB")  # Convert to RGB
    
    st.image(payload_image, caption="Payload Image (grayscale)", width=200)
    st.image(carrier_image, caption="Carrier Image (RGB)", width=200)

    # Preprocess Images
    payload_input = preprocess_image(payload_image, (32, 32))  # Resizing payload
    carrier_input = preprocess_image(carrier_image, (32, 32))  # Resizing carrier
    
    # Encode Section
    st.header("2. Encoding")
    if st.button("Encode Payload into Carrier"):
        # Use the steganography model to encode
        stego_image = steganography_model.predict([carrier_input, payload_input])
        stego_image = (stego_image[0] * 255).astype(np.uint8)  # Rescale to [0, 255]
        st.image(stego_image, caption="Stego Image (Encoded)", width=200)

        # Decode Section
        st.header("3. Decoding")
        if st.button("Decode Payload from Stego Image"):
            # Use the decoder to extract the payload
            decoded_payload = steganography_model.predict([stego_image / 255.0, np.zeros_like(payload_input)])
            decoded_payload = (decoded_payload[0, :, :, 0] * 255).astype(np.uint8)  # Rescale
            st.image(decoded_payload, caption="Decoded Payload", width=200)

else:
    st.info("Please upload both payload and carrier images.")

st.markdown("---")
st.write("Made with ❤️ using Streamlit.")
