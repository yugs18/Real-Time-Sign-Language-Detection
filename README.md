# Real-Time Sign Language Detection

## Project Overview
The **Real-Time Sign Language Detection** project aims to create a model that can recognize and classify various sign language gestures in real-time. This involves capturing training data, processing images, training a model, and performing real-time detection through a webcam.

---

## Data Collection

### Overview
The data collection script captures images of specific sign language gestures using a webcam. It organizes these images into folders by gesture name and generates unique filenames for each image to ensure a consistent dataset for training the model.

### Code Description

1. **Library Imports**:
   - `cv2`: Used for capturing and processing video frames.
   - `os`: For creating directories to store images.
   - `time`: Manages delays in capture timing.
   - `uuid`: Generates unique IDs for each image filename.

2. **Configurations**:
   - `IMAGES_PATH`: Sets the main directory to save all collected images.
   - `labels`: Defines gestures to capture, with each label corresponding to a unique gesture.
   - `number_of_imgs`: The number of images captured per gesture.

3. **Functions**:
   - `create_directory(path)`: Checks if a specified path exists; if not, it creates the directory to store images.
   - `capture_images_for_class(label, num_images, video_capture)`: Captures `num_images` images for each gesture label, storing each image with a unique UUID-based filename.
   - `main()`: Initializes the main data directory and webcam, iterating through each gesture to capture the specified number of images.

4. **Execution**:
   - The script is designed to run directly (`__name__ == "__main__"`), initializing the collection process and properly handling webcam resources to avoid conflicts with other applications.

### Usage

1. Run the script, ensuring the webcam is enabled.
2. For each gesture, the script waits 5 seconds to allow for preparation and then captures the specified number of images.
3. Images are saved in folders named after each gesture label under the `IMAGES_PATH` directory.

---

## Data Labeling

### Overview
After collecting gesture images, Label Studio was used to annotate each image with the appropriate gesture label. This labeled data provides essential training information for the model.

### Steps

1. **Labeling**:
   - Imported the collected images into Label Studio for annotation.
   - Assigned gesture labels corresponding to the captured images to enable precise model training.

2. **Exporting**:
   - The labeled images were exported in **YOLO format**, making them compatible with common object detection frameworks that use YOLO-based models for training.

---

## Training

### Overview
After data collection and labeling, the model was trained using a YOLO-based object detection approach. This stage uses the YOLOv5 library, specifically customized for recognizing sign language gestures based on the labeled data.

### Code Description

1. **Training Configuration**:
   - The model is trained with the following specifications:
     - **Task**: Object detection (`detect`)
     - **Model**: Custom YOLO configuration (`yolo11s.pt`)
     - **Data**: Data configuration file for training (e.g., `data.yaml`)
     - **Epochs**: 50, allowing for sufficient training cycles for gesture recognition
     - **Image Size**: 640x640 for both training and validation images
   - Training output is stored under `runs/detect/train`, where the best model weights are saved as `best.pt`.

2. **Model Loading**:
   - Loads the best-performing model weights after training for inference.

3. **Inference**:
   - The trained model performs predictions on validation images (`val` set).
   - Predictions are filtered based on a confidence threshold of `0.25`.
   - Each prediction result is displayed with bounding boxes and gesture labels, helping verify model accuracy.

---

## Note

The model was trained on a relatively small dataset and with the lightweight YOLOv5 variant, `yolo11s.pt`. Due to the limited data and the simplified model configuration, overall accuracy may be limited. Further improvements could be achieved by expanding the dataset and adjusting model parameters.

--- 
