# Fashion MNIST Image Classification Project

This project implements a Convolutional Neural Network (CNN) to classify fashion items using the Fashion MNIST dataset. It includes both a model training notebook and a web application for making predictions on new images.

## Project Structure
```
├── Fashion_mnist_model_training.ipynb   # Jupyter notebook for model training
├── app.py                              # Streamlit web application
├── fashion_mnist_model.h5              # Trained model file
├── requirements.txt                    # Project dependencies
└── README.md                          # Project documentation
```

## Features

- CNN model with data augmentation and batch normalization
- Interactive web interface for image upload and prediction
- Real-time predictions with confidence scores
- Visualization of prediction probabilities
- Support for 10 fashion item categories

## Model Architecture

The CNN architecture includes:
- Multiple convolutional layers with increasing filters (32 -> 64 -> 128)
- Batch normalization for training stability
- Dropout layers for preventing overfitting
- Dense layers with ReLU and softmax activation

## Categories

The model can classify the following fashion items:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Setup and Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Train the model (optional, pre-trained model included):
- Open and run `Fashion_mnist_model_training.ipynb`

3. Run the web application:
```bash
streamlit run app.py
```

## Usage

1. Launch the web application
2. Upload an image of a fashion item
3. View the prediction and confidence scores
4. Explore the probability distribution across all categories

## Model Performance

- Training includes data augmentation for better generalization
- Early stopping to prevent overfitting
- Achieves high accuracy on the test set
- Uses image preprocessing for better real-world performance

## Technologies Used

- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib
