# Name- Raj chouhan , Email- rajchouhan1042@gmail.com, Project name is - Implementation of ML model for image classification.
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")

    uploaded_file = st.file_uploader("Upload an Image (jpg/png)", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        model = tf.keras.applications.MobileNetV2(weights="imagenet")

        
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        st.subheader("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. {label}: {score * 100:.2f}%")


def resnet50_imagenet():
    st.title("Image Classification with ResNet50")

    uploaded_file = st.file_uploader("Upload an Image (jpg/png)", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        model = tf.keras.applications.ResNet50(weights="imagenet")

        
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0]

        st.subheader("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. {label}: {score * 100:.2f}%")


def cifar10_classification():
    st.title("CIFAR-10 Image Classification")

    uploaded_file = st.file_uploader("Upload an Image (jpg/png)", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        model = tf.keras.models.load_model("model111.h5")

        
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        
        img = image.resize((32, 32))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.subheader("Results:")
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

def about_page():
    st.title("About This Application")
    st.write("""
        Welcome to this powerful image classification app! 
        This application allows users to classify images using state-of-the-art deep learning models:
        
        - **MobileNetV2**: Lightweight and fast model trained on ImageNet.
        - **ResNet50**: Robust and accurate model for large-scale image classification.
        - **CIFAR-10**: Custom-trained model for classifying 10 object categories (e.g., airplane, cat, dog).
        
        **Author**: Raj chouhan 
        **Contact**: [rajchouhan1042@gmail.com](mailto:rajchouhan1042@gmail.com)
            
        This project, Implementation of an ML Model for Image Classification, addresses the challenge of accurate and efficient categorization of images into predefined classes. The problem stems from the increasing volume of visual data and the need for automated systems capable of extracting meaningful insights.
            The primary objective of this project is to develop a machine learning model, specifically leveraging Convolutional Neural Networks (CNNs), to classify images with high precision and minimal error rates. The methodology includes preprocessing the dataset through normalization and augmentation, designing a CNN architecture tailored for the task, and training the model using a supervised learning approach. Tools such as TensorFlow and Python were employed to implement the solution.
            Key results demonstrated the efficacy of the model, achieving an accuracy of over 90% on the CIFAR-10 dataset. Performance metrics such as precision, recall, and F1-score further validate the robustness of the approach.
            The conclusion highlights the potential of CNN-based image classification in applications like object detection, medical diagnostics, and automated surveillance. Suggestions for future work include expanding the dataset, optimizing hyperparameters, and exploring advanced techniques such as transfer learning for improved performance.
            This project underscores the relevance of machine learning in solving real-world challenges and serves as a foundation for further research in the domain of computer vision.

    """)


def footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>© 2024 Raj Chouhan | <a href="mailto:rajchouhan1042@gmail.com">rajchouhan1042@gmail.com</a></p>
            <a href="https://www.facebook.com/rokey.chintu.1/" target="_blank" style="margin-right: 15px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook" style="width: 24px; height: 24px;">
            </a>
            <a href="https://www.linkedin.com/in/raj-chouhan-b16042327/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" alt="LinkedIn" style="width: 24px; height: 24px;">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ("Home", "About"))

    if choice == "Home":
        st.sidebar.title("Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose a Model",
            (
                "MobileNetV2 (ImageNet Classification)",
                "ResNet50 (ImageNet Classification)",
                "CIFAR-10 Image Classification",
            )
        )

        if model_choice == "MobileNetV2 (ImageNet Classification)":
            mobilenetv2_imagenet()
        elif model_choice == "ResNet50 (ImageNet Classification)":
            resnet50_imagenet()
        elif model_choice == "CIFAR-10 Image Classification":
            cifar10_classification()
    elif choice == "About":
        about_page()

    
    footer()

if __name__ == "__main__":
    main()
