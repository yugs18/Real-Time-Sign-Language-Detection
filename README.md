# Real-Time Sign Language Detection

# Abstract

The Real-Time Sign Language Detection project aims to create an accessible communication tool for the hearing impaired using computer vision and machine learning. This system recognizes sign language gestures in real-time through video input, focusing on developing a robust gesture recognition model, ensuring seamless processing, and providing a user-friendly interface.

Key components include data collection via webcam images, image processing with bounding box annotations, and dataset preparation for model training. A Convolutional Neural Network (CNN) is built with TensorFlow and Keras, incorporating data augmentation for improved performance. The trained model is evaluated on accuracy and loss metrics and deployed for real-time gesture recognition.

This project enhances communication accessibility, facilitating interactions between the hearing and hearing-impaired communities.

---

# Introduction

The Real-Time Sign Language Detection project aims to create an efficient system for recognizing sign language gestures using computer vision and machine learning techniques. This project addresses the growing need for accessible communication tools, particularly for the hearing impaired, by enabling real-time interpretation of sign language through video input.

### Objectives
- **Gesture Recognition**: Develop a model capable of accurately detecting and interpreting various sign language gestures.
- **Real-Time Processing**: Implement a system that processes video streams from a webcam, providing instant feedback and predictions.
- **User-Friendly Interface**: Create an intuitive experience for users to capture and interact with gesture data seamlessly.

### Key Components
1. **Data Collection**: Utilize a webcam to capture images of different sign language gestures, which are stored with unique identifiers for effective training.
2. **Image Processing**: Process the collected images to draw bounding boxes around detected hands and label them accordingly, preparing the dataset for model training.
3. **Model Development**: Build a Convolutional Neural Network (CNN) using TensorFlow and Keras, incorporating data augmentation techniques to enhance model robustness and accuracy.
4. **Training and Evaluation**: Train the model with a carefully split dataset of labeled images and evaluate its performance using accuracy and loss metrics.
5. **Deployment**: Implement the model in a real-time application, allowing users to perform sign language gestures and receive immediate recognition feedback.

---

## Data Collection

### Overview
Captures images of sign language gestures for training the detection model.

### Key Components
- **Libraries**: Utilizes `cv2` (webcam access), `os` (directory management), `time` (delays), `uuid` (unique filenames).
- **Constants**: Defines image save paths and gesture labels, along with the number of images to capture.

### Functions
- **Directory Creation**: Ensures necessary folders exist for saving images.
- **Image Capture**: Captures and saves images with unique names.

### Main Process
Initializes the webcam and manages data collection.

---

## Image Processing and Labeling

### Overview
Processes collected images to label gestures with bounding boxes for model training.

### Key Components
- **Libraries**: `cv2`, `os`, `mediapipe` (hand tracking), `IPython.display` (image display).
- **Paths**: Defines directories for raw and labeled images.

### Functions
- **MediaPipe Setup**: Initializes hand-tracking for gesture detection.
- **Bounding Box Drawing**: Detects hands and annotates images.
- **Image Processing**: Iterates through raw images, applies bounding boxes, and saves labeled versions.

---

## Dataset Preparation

### Overview
Organizes images into training and validation sets for model training.

### Key Components
- **Imports**: Uses `os`, `shutil`, `random` for file management and shuffling.
- **Paths**: Defines source and target directories for the dataset.

### Process
- **Directory Creation**: Creates folders for training and validation data.
- **Image Shuffling**: Randomly divides images into training (90%) and validation (10%) sets.

---

## Model Training

### Overview
Builds and trains a CNN for sign language gesture detection.

### Key Components
- **Libraries**: Uses TensorFlow and Keras for model development.
- **Data Augmentation**: Implements transformations for training images to enhance dataset diversity.

### Process
- **Loading Data**: Generates batches of image data with labels.
- **Model Architecture**: Defines layers (convolutional, pooling, dense) and compiles the model.
- **Training**: Fits the model with training data and tracks performance metrics.

### Finalization
- **Model Saving**: Saves the trained model for future use.
- **Visualization**: Plots accuracy and loss over epochs for performance analysis.

---

# Real-Time Sign Language Detection

## Importing Necessary Libraries
1. **OpenCV**: For video processing, capturing from the webcam, and drawing bounding boxes.
2. **OS Module**: Tools for file management and OS interaction.
3. **NumPy**: For numerical operations and array handling.
4. **TensorFlow**: A framework for building and training neural networks.
5. **Keras Preprocessing**: Functions like `img_to_array` for converting images to NumPy arrays.
6. **MediaPipe**: For real-time hand tracking in video streams.

## Model Loading and Label Definitions
- Load the trained sign language detection model and define labels for each gesture as dictionaries containing name and ID.

## Setting Up MediaPipe for Hand Detection
- Initialize MediaPipe hands module with parameters for detection and tracking confidence.

## Label Retrieval Function
- Define a function to map label IDs to names by iterating through the label list.

## Webcam Initialization
- Activate the webcam using OpenCV for real-time capture.

## Main Loop for Video Processing
- Continuously read frames and process them for hand detection using MediaPipe.
- For detected hands:
  - Initialize bounding box coordinates and analyze landmarks to determine dimensions.
  - Draw landmarks on the frame.

## Hand Image Extraction and Prediction
- Extract the ROI for the detected hand, resize and normalize it.
- Make predictions with the loaded model, displaying the predicted label and confidence on the frame.

## Displaying the Results
- Show the processed frame with bounding box and prediction text, continuing until 'q' is pressed.

## Cleanup Operations
- Release the webcam and close all OpenCV windows after exiting the loop.

