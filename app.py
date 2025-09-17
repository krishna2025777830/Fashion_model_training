import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28 using LANCZOS resampling for better quality
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    # Convert to numpy array and invert if needed (Fashion MNIST has white digits on black background)
    img_array = np.array(image)
    mean_pixel = np.mean(img_array)
    if mean_pixel > 127:
        img_array = 255 - img_array
    # Normalize to [0,1]
    img_array = img_array.astype('float32') / 255.0
    # Add contrast
    img_array = np.clip((img_array - img_array.mean()) * 1.5 + img_array.mean(), 0, 1)
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Set up the Streamlit page
st.title('Fashion Item Classifier')
st.write('Upload an image of a fashion item, and I will try to identify it!')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    probabilities = tf.nn.softmax(predictions[0])
    predicted_class = np.argmax(probabilities)
    confidence = float(probabilities[predicted_class])
    
    # Set confidence threshold
    CONFIDENCE_THRESHOLD = 0.7
    
    # Check image quality and model confidence
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("🤔 I don't know this image, boss! This doesn't look like any fashion item I was trained on.")
        st.write("I can only recognize these fashion items:")
        for item in class_names:
            st.write(f"- {item}")
        # Add helpful tip for unknown images
        st.info("💡 Tip: Try uploading a clear image of one of the fashion items listed above!")
    else:
        # Display results with emoji based on confidence
        if confidence > 0.9:
            emoji = "🎯"
        elif confidence > 0.8:
            emoji = "👍"
        else:
            emoji = "🤔"
            
        st.success(f"{emoji} I think this is a **{class_names[predicted_class]}**!")
        st.write(f"Confidence: {confidence*100:.2f}%")
        
        # Only show probability distribution for recognized images
        st.subheader("Probability Distribution")
        probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        # Sort probabilities in descending order
        sorted_probs = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
        st.bar_chart(sorted_probs)
