# Sign Language Gesture Recognition using Deep Learning

This project implements a sign language gesture recognition system using two different approaches: one based on images (using ResNet50) and another based on hand keypoints (using LSTM). The system can be used for both training models and making real-time predictions on hand gestures captured via a webcam.

---

## Abstract

The ability to recognize sign language gestures through machine learning is a powerful tool that can bridge communication gaps between individuals who are deaf or hard of hearing and those who do not understand sign language. This project aims to develop a real-time sign language gesture recognition system using deep learning techniques. Two distinct approaches are employed: an image-based approach using a pre-trained ResNet50 model for gesture classification, and a keypoint-based approach utilizing LSTM (Long Short-Term Memory) networks to process hand gesture sequences based on 3D keypoint data extracted through the MediaPipe framework. The dataset for this project consists of labeled images of gestures, categorized into five classes: "hello", "thanks", "yes", "no", and "iloveyou". Both models are trained to predict these gestures in real-time from webcam feed, offering a versatile solution for gesture recognition across various use cases, including real-time communication, accessibility, and assistive technology for the hearing impaired.

---

## Introduction

Sign language serves as the primary mode of communication for millions of people around the world who are deaf or hard of hearing. Despite its significance, a gap often exists in communication between sign language users and individuals who are not familiar with it. Machine learning and computer vision provide promising solutions to this problem by enabling real-time recognition and translation of hand gestures into readable text or speech. In this project, we explore two different approaches to recognize sign language gestures: the image-based approach and the keypoint-based approach.

The **image-based approach** utilizes the ResNet50 convolutional neural network (CNN), a pre-trained deep learning model, to classify images of hand gestures. This model learns the features of hand gestures by processing labeled gesture images, making it well-suited for real-time gesture recognition via webcam. The **keypoint-based approach**, on the other hand, leverages the power of LSTM networks to process 3D keypoints of the hand, which are extracted using Google's MediaPipe framework. These keypoints represent the positions of different hand joints, providing a more detailed understanding of the hand's movement over time, which is crucial for recognizing dynamic gestures.

Both methods allow for real-time predictions, with the webcam feed acting as the input for gesture recognition. This project demonstrates the effectiveness of both models, showing how machine learning can be used to facilitate better communication and accessibility for sign language users. By combining computer vision and deep learning, this project offers a step towards bridging communication barriers and creating more inclusive environments.

---

## Accuracy Results

In the evaluation of both approaches, the **image-based approach** using the ResNet50 model achieved an accuracy of approximately **20%**. This relatively low accuracy indicates that the image-based method may struggle with recognizing gestures due to factors such as varying lighting conditions, hand orientations, and background noise.

On the other hand, the **keypoint-based approach** utilizing the LSTM model, which processes 3D hand keypoints extracted via MediaPipe, performed significantly better. This approach achieved an accuracy of over **98%**, demonstrating its superior capability in recognizing sign language gestures. The keypoint-based method benefits from capturing detailed hand movements, making it more robust and effective, especially for dynamic gestures and real-time recognition.

These results highlight the effectiveness of using hand keypoints over image data for gesture recognition, particularly in real-time applications.

---

## Approaches

### Approach 1: Using ResNet50 (Image-based)

This approach uses a pre-trained ResNet50 model for hand gesture classification based on images.

#### Steps:
1. **Collecting Data**:
   - Capture images of hand gestures using a webcam. Each gesture should be labeled accordingly (e.g., "hello", "thanks", etc.).
   
2. **Labeling**:
   - Label the captured images with bounding boxes around the hands. This can be done manually or using a hand detection method.
   
3. **Splitting the Data**:
   - Split the dataset into training and validation sets. This can be done using an 80-20 split or as per your preference.

4. **Training**:
   - Train a model using the ResNet50 architecture for image classification. Data augmentation techniques such as rotation, zoom, and flips can be applied to enhance the model's performance.

5. **Real-Time Prediction**:
   - Use the trained model to make real-time predictions on hand gestures using a webcam feed. The system will predict the gesture based on the live video and display the result.

### Approach 2: Using LSTM (Keypoint-based)

This approach focuses on extracting 3D keypoints of hand gestures using MediaPipe and training an LSTM model for gesture recognition.

#### Steps:
1. **Collecting Data**:
   - Capture hand gestures in front of the webcam. MediaPipe will be used to extract 3D keypoints of the hand.

2. **Keypoint Extraction**:
   - Extract the 3D keypoints (x, y, z coordinates) of the hand from each gesture frame. Save these keypoints to CSV files, with each gesture having its own file (e.g., `hello_keypoints.csv`).

3. **Data Preprocessing**:
   - Preprocess the extracted keypoints by normalizing and padding the sequences to ensure uniform input size for the LSTM model.

4. **Training**:
   - Train an LSTM model with the preprocessed keypoints data. The model should use bidirectional LSTM layers for better performance on sequential data.

5. **Real-Time Prediction**:
   - Use the trained LSTM model to predict hand gestures in real-time from the webcam feed. The system will use the extracted keypoints to make predictions and display the gesture label.


---

# Sign Language Gesture Image Collection

This script is designed to capture images for training a machine learning model to recognize basic sign language gestures. It uses a webcam to capture images and saves them into separate folders for each gesture.

## Features

- Captures a specified number of images per gesture label.
- Organizes images in labeled directories.
- Uses unique filenames for each captured image.

## Setup

1. **Install Dependencies**:
   - Ensure `opencv-python`, `os`, `time`, and `uuid` libraries are installed.

2. **Run the Script**:
   - Execute the script directly. Press `q` to stop early.

## Labels and Usage

Each label represents a gesture, such as "hello" or "thanks." Customize labels and image count as needed.

---

# Sign Language Gesture Image Labeling

This project adds bounding boxes and labels to hand gesture images to aid in training a sign language detection model. It uses MediaPipe for hand tracking and labels gestures based on predefined categories.

## Features

- Draws bounding boxes around hands and labels each gesture.
- Saves processed images in a specified directory for training.

## Setup

1. **Install Dependencies**:
   - `opencv-python`, `mediapipe`, and `IPython.display` libraries are required.

2. **Prepare Data**:
   - Place raw images in `images/collected` directory with folders named after each gesture label.

3. **Run the Script**:
   - Execute the script directly. Processed images will be saved in `images/labeled`.

## Usage Notes

- Customize labels and directory paths as needed.
- Images are displayed during processing in Notebook environments.

---

# Sign Language Gesture Keypoint Extraction

This project extracts 3D hand landmark coordinates from images of sign language gestures using the MediaPipe library. Extracted keypoints are saved in CSV files for each gesture, aiding in training machine learning models for gesture recognition.

## Features

- Extracts 21 3D hand landmarks (x, y, z coordinates) for each detected hand.
- Saves keypoint data to CSV files, with each row representing a labeled image.

## Setup

1. **Install Dependencies**:
   - Required libraries: `opencv-python`, `mediapipe`, and `csv`.

2. **Prepare Data**:
   - Place gesture images in `images/collected` directory, organizing them into subfolders named after each gesture label (e.g., "hello", "thanks").

3. **Run the Script**:
   - Execute the script directly. Keypoint data will be saved in the `images/keypoints` directory, organized by gesture labels.

## Code Structure

- **Paths and Labels**: Configurable paths and gesture labels.
- **Keypoint Extraction**: Detects hand landmarks in each image.
- **CSV Output**: Saves the keypoints in CSV format, with each image's landmarks stored in a new row.

## Notes

- Ensure gesture images are in RGB format for optimal landmark detection.
- Processed images should be static for accurate keypoint extraction.

---

# Sign Language Gesture Dataset Splitter

This project organizes a dataset of labeled images of sign language gestures into training and validation sets. Images are randomly shuffled and split according to a specified ratio, enabling separate directories for training and validation data.

## Features

- Splits images by label into separate `train` and `validation` directories.
- Supports customizable labels and split ratios.

## Setup

1. **Install Dependencies**:
   - Required libraries: `os`, `shutil`, and `random`.

2. **Prepare Data**:
   - Place labeled images in `images/labeled`, with each gesture in its own subfolder, such as `images/labeled/hello`, `images/labeled/thanks`, etc.

3. **Specify Directories and Parameters**:
   - Update `SOURCE_DIR` with the path to the labeled images.
   - Define `TARGET_BASE_DIR` to specify where split data will be saved.
   - Adjust `train_ratio` to set the proportion of images for training.

4. **Run the Script**:
   - Execute the script to organize the dataset. Training and validation images will be saved in directories within the `TARGET_BASE_DIR`.

## Code Structure

- **Paths and Labels**: Configurable source and target paths, along with label definitions.
- **Data Splitting**: Randomly shuffles images for each label and splits them by `train_ratio`.
- **Data Organization**: Moves images to separate `train` and `validation` directories based on split.

## Notes

- Ensure gesture images are organized by label in the source directory.
- Use an appropriate split ratio to balance training and validation data.

---

# Sign Language Gesture Detection Model

This project provides a deep learning model for detecting hand gestures used in sign language. Using transfer learning with the pre-trained ResNet50 model, the model is fine-tuned to classify gestures such as "hello", "thanks", "yes", "no", and "I love you". The model uses image augmentation techniques for the training dataset and evaluates its performance on a validation set.

## Features

- **Transfer Learning**: Utilizes the pre-trained ResNet50 model as the base and fine-tunes it for gesture recognition.
- **Data Augmentation**: Enhances the training dataset using various augmentation techniques such as rotation, zoom, and flips.
- **Custom Callback**: Includes a callback to stop training once 92% accuracy is achieved.
- **Learning Rate Scheduler**: Automatically adjusts the learning rate if validation accuracy plateaus.
- **Evaluation**: Computes performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Setup

1. **Install Dependencies**:
   - Required libraries: `tensorflow`, `scikit-learn`, `matplotlib`, `numpy`
   - Ensure that TensorFlow is installed (`pip install tensorflow`) for model training.

2. **Prepare Dataset**:
   - Organize the dataset in the following structure:
     ```
     data/
         train/
             hello/
             thanks/
             yes/
             no/
             iloveyou/
         validation/
             hello/
             thanks/
             yes/
             no/
             iloveyou/
     ```
   - The images in each folder should represent the respective gestures.

3. **Configure Parameters**:
   - Adjust paths for the training and validation datasets in `train_data_dir` and `validation_data_dir`.
   - Modify `img_height`, `img_width`, and `batch_size` as needed to suit your dataset and hardware capabilities.

4. **Model Configuration**:
   - The model is based on the ResNet50 architecture, with a few custom layers for fine-tuning. You can modify the architecture by adding/removing layers.

## Training

- The model uses `categorical_crossentropy` as the loss function for multi-class classification and tracks accuracy as the evaluation metric.
- The model is trained using the Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate if validation accuracy stagnates.
- Training can be stopped early using the custom `StopAtAccuracy` callback when the accuracy exceeds 92%.

## Model Evaluation

- **Metrics**: After training, the model's performance is evaluated using accuracy, F1-score, precision, recall, and a classification report.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the model's performance across all classes.

## Results Visualization

- Training and validation accuracy and loss are plotted after training to visualize the model's learning progress.

## Saving the Model

- The trained model is saved in Keras format (`.keras`) to disk after training is complete.

## Usage

1. **Load the Saved Model**:
   - You can load the trained model using `tf.keras.models.load_model('sign_language_detection_model.keras')`.
   
2. **Make Predictions**:
   - Use the model to predict hand gestures on new images by passing them through `model.predict()`.

## Notes

- Ensure that the images are correctly preprocessed before training (resizing and normalization).
- Adjust the learning rate and batch size for better performance based on the size of your dataset and hardware.

---

# Real-Time Sign Language Gesture Recognition

This project implements a real-time sign language gesture recognition system using a pre-trained deep learning model for hand gesture classification. The system leverages OpenCV for video processing, MediaPipe for hand tracking, and TensorFlow for model inference to recognize sign language gestures through a webcam feed.

## Features

- **Hand Detection**: Uses MediaPipe to detect hand landmarks and track hand movements in real time.
- **Real-Time Classification**: The model classifies hand gestures as one of the predefined classes (e.g., "hello", "thanks", "yes", "no", "I love you").
- **Bounding Box & Confidence**: Displays a bounding box around the detected hand along with the predicted gesture and confidence score.
- **Model Inference**: Utilizes a trained deep learning model (`sign_language_model.h5`) to predict gestures based on detected hand images.

## Setup

1. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install opencv-python mediapipe tensorflow numpy
     ```
   - These libraries are used for video processing, hand tracking, machine learning, and numerical operations.

2. **Load Trained Model**:
   - The system uses a pre-trained model (`sign_language_model.h5`) for hand gesture classification. You must ensure this model file is available in the same directory or specify its path.
   - If you have trained your own model, save it in `.h5` format and replace the model file name.

3. **Set up Labels**:
   - The labels for the hand gestures are defined in the `labels` list, where each gesture is associated with a unique ID. Modify this list to fit your gesture classes.

4. **Webcam Access**:
   - The system accesses the default webcam (`cv2.VideoCapture(0)`). Ensure your webcam is connected and functional.

## How It Works

1. **Hand Detection**:
   - MediaPipe processes each video frame to detect hand landmarks. The system calculates a bounding box around the detected hand for further processing.

2. **Image Preprocessing**:
   - The hand region of interest (ROI) is extracted, resized to match the model’s input size, and normalized before being fed into the trained model.

3. **Model Prediction**:
   - The preprocessed image is passed to the TensorFlow model (`sign_language_model.h5`) for classification. The predicted class and confidence score are displayed on the screen in real-time.

4. **Real-Time Display**:
   - A bounding box is drawn around the detected hand, and the predicted gesture label with the confidence score is shown on the screen.

## Running the Application

1. **Start the Program**:
   - Run the Python script to start the real-time gesture recognition system. The webcam feed will open, and the system will continuously detect and classify hand gestures.

2. **Exit the Program**:
   - To exit the program, press the `q` key while the video feed window is active.

## Model Details

- **Input**: 150x150 RGB images of hand gestures.
- **Output**: A predicted class label corresponding to one of the defined gestures, along with the confidence score.
- **Model Format**: The model is saved in `.h5` format and loaded using TensorFlow's `load_model()` function.

## Label Mapping

- The predefined gestures are mapped to specific IDs:
  - `1` -> "hello"
  - `2` -> "thanks"
  - `3` -> "yes"
  - `4` -> "no"
  - `5` -> "I love you"

The `get_label_name` function maps the predicted label ID to its corresponding gesture name for display purposes.

## Notes

- **Performance**: The performance of the system depends on the processing power of your machine and the quality of the webcam feed.
- **Model Accuracy**: The model's accuracy depends on the training data quality and the gestures in your dataset.
- **Webcam Quality**: The quality of hand detection and classification can vary based on the resolution and frame rate of the webcam.

## Troubleshooting

- **No Hand Detection**: Ensure that your hand is clearly visible in the camera's field of view. Adjust the lighting or camera angle if necessary.
- **Low Accuracy**: If the model is providing incorrect predictions, it may require retraining with more diverse and representative data.
- **Error Messages**: Ensure all dependencies are correctly installed and that the model file is located in the specified path.

---

# Sign Language Gesture Recognition using LSTM

This project implements a gesture recognition system using an LSTM-based model to classify sign language gestures. The system utilizes keypoints data extracted from video frames and trains a deep learning model to classify gestures such as "hello", "thanks", "yes", "no", and "I love you".

## Features

- **LSTM Model**: The system uses a Long Short-Term Memory (LSTM) architecture with Bidirectional layers for classifying sign language gestures.
- **Keypoints Data**: The input data is based on keypoints extracted from sign language gestures.
- **Data Preprocessing**: The code handles loading, preprocessing, and encoding the dataset of keypoints.
- **Model Evaluation**: The system evaluates the model on a validation set, with accuracy and loss metrics plotted, and a classification report and confusion matrix generated.

## Setup

1. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install numpy pandas tensorflow scikit-learn seaborn matplotlib
     ```

2. **Prepare Dataset**:
   - The dataset should contain CSV files for each gesture (e.g., `hello_keypoints.csv`, `thanks_keypoints.csv`) in a directory called `data`. 
   - Each CSV file should include keypoints data representing the corresponding gesture. The files should not contain an image name column, only the keypoints data.

3. **Data Structure**:
   - The keypoints data is expected to be stored in a format where each row contains keypoints for a sample. The `data` directory should contain the following files:
     - `hello_keypoints.csv`
     - `thanks_keypoints.csv`
     - `yes_keypoints.csv`
     - `no_keypoints.csv`
     - `iloveyou_keypoints.csv`

## How It Works

1. **Loading and Preprocessing Data**:
   - The code loads the keypoints data for each gesture, encodes the labels using `LabelEncoder`, and concatenates all keypoints into a single dataset.
   - The sequences of keypoints are padded to ensure uniform length for LSTM input.

2. **Model Architecture**:
   - The model consists of two Bidirectional LSTM layers, followed by Dense layers for classification. Dropout layers are added to reduce overfitting.

3. **Model Training**:
   - The model is trained on the data using the Adam optimizer and sparse categorical cross-entropy loss. It runs for 30 epochs, and both training and validation accuracy and loss are tracked.

4. **Model Evaluation**:
   - After training, the model is evaluated using a classification report and confusion matrix, which display detailed performance metrics for each gesture class.

5. **Model Saving**:
   - Once trained, the model is saved to a file named `sign_language_model.keras` for future use.

## Running the Application

1. **Start the Training**:
   - Run the Python script to train the model. The system will load the keypoints data, train the model, and display training/validation accuracy and loss curves.

2. **Model Evaluation**:
   - After training, the model’s performance is evaluated on the validation set. A classification report and confusion matrix are displayed, providing insights into the model’s classification accuracy for each gesture.

3. **Saving the Model**:
   - After training and evaluation, the model is saved to the `sign_language_model.keras` file, which can later be used for inference in real-time applications.

## Model Details

- **Input**: The input to the model is a sequence of keypoints data for each gesture.
- **Output**: The model outputs a classification for each gesture, such as "hello", "thanks", etc.
- **Architecture**: 
  - Bidirectional LSTM layers with 64 and 32 units.
  - Dense layers with 64 units and ReLU activation.
  - Dropout layers to reduce overfitting.
  - Softmax output layer for multi-class classification.

## Metrics

- **Accuracy**: Tracks how well the model performs on both training and validation sets.
- **Loss**: Tracks the model’s loss during training.
- **Classification Report**: Provides precision, recall, and F1-score for each gesture class.
- **Confusion Matrix**: Visualizes the true vs. predicted labels.

## Notes

- **Performance**: The performance depends on the quality of keypoints data and the architecture of the model. You may need to experiment with different parameters for optimal results.
- **Data Quality**: Ensure the keypoints data is clean and properly formatted for accurate results.
- **Overfitting**: Dropout layers are included to mitigate overfitting, but further fine-tuning may be necessary based on your dataset size and variability.

---

# Real-Time Sign Language Recognition

This project enables real-time sign language recognition using a webcam. The system utilizes a pre-trained deep learning model to classify hand gestures, displaying the recognized sign language gesture (e.g., "hello", "thanks", "yes") in real-time on the screen. The keypoints of the hand, detected using MediaPipe, are fed into the model to make predictions.

## Features

- **Real-Time Gesture Recognition**: Classifies sign language gestures in real-time using a webcam.
- **Hand Detection**: Uses MediaPipe's hand detection to locate hand landmarks and extract keypoints.
- **Gesture Classification**: A pre-trained deep learning model classifies the extracted hand gestures into predefined labels.
- **Webcam Interface**: Captures video feed from the webcam and displays the predicted gesture on the screen.
- **Easy Exit**: Press 'q' to exit the application at any time.

## Setup

1. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install numpy opencv-python mediapipe tensorflow
     ```

2. **Download Pre-trained Model**:
   - Ensure that the pre-trained model `sign_language_model.keras` is available in the project directory.

3. **Webcam Setup**:
   - The system uses the default webcam (camera index 0). Ensure that the webcam is properly connected and accessible.

## How It Works

1. **MediaPipe Hand Detection**:
   - The webcam feed is processed frame by frame using MediaPipe's hand detection module.
   - The detected hand landmarks are extracted and converted into keypoints.

2. **Prediction**:
   - The extracted hand keypoints are reshaped and passed into the pre-trained model (`sign_language_model.keras`).
   - The model outputs the predicted gesture class, which is mapped to a predefined label.

3. **Display Results**:
   - The predicted label is displayed on the screen above the detected hand landmarks.
   - The webcam feed is displayed in a window, and predictions are continuously updated in real-time.

4. **Exit the Application**:
   - Press 'q' to exit the application and close the webcam feed.

## Model Details

- **Model Input**: The model takes in the flattened hand landmarks (keypoints) extracted from the hand in the video frame.
- **Model Output**: The model outputs a predicted gesture class, which is mapped to one of the predefined labels.
- **Supported Labels**:
  - "hello"
  - "iloveyou"
  - "no"
  - "thanks"
  - "yes"

## Running the Application

1. **Start the Application**:
   - Run the Python script to start real-time gesture recognition using the webcam.
   - The system will continuously process the video feed and display the predicted gesture.

2. **Test the Model**:
   - Perform sign language gestures in front of the webcam. The system will recognize and display the predicted gesture on the screen.

3. **Exit**:
   - Press 'q' to stop the real-time prediction and close the webcam window.

## Notes

- **Model Compatibility**: Ensure the model `sign_language_model.keras` is trained with the same hand landmark data and has the correct output labels.
- **Hand Detection**: MediaPipe's hand detection works best when hands are clearly visible in the frame and within the camera's field of view.
- **Performance**: The frame rate and performance may vary depending on the computer's processing power and camera quality.

---