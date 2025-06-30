# Age Detection System using Deep Learning

## Overview

This Age Detection System uses deep learning models integrated with OpenCV to detect human faces and predict their age group from images or video streams in real time. It is optimized for both performance and accuracy, making it suitable for hackathon demos, academic projects, or real-world applications.

---

## Features

* 👤 **Face Detection** using OpenCV DNN module
* ⏳ **Age Prediction** using pre-trained Caffe models
* 📸 Supports both image and real-time webcam input
* ✅ Displays bounding boxes with age labels and confidence
* ⚖️ Performance stats like FPS and inference time (for video)

---

## Age Groups Predicted

* (0-2)
* (4-6)
* (8-12)
* (15-20)
* (25-32)
* (38-43)
* (48-53)
* (60-100)

---

## Folder Structure

```
AgeDetectionSystem/
├── models/
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
├── snapshots/
├── output/
├── age_detection.py
└── README.md
```

---

## Requirements

* Python 3.6+
* OpenCV (with DNN module)
* NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Pre-trained Models

You need to download the following models:

### 1. Face Detection (TensorFlow)

* `opencv_face_detector.pbtxt`
* `opencv_face_detector_uint8.pb`

### 2. Age Prediction (Caffe)

* `age_deploy.prototxt`
* `age_net.caffemodel`

You can find these in OpenCV's [GitHub model zoo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) or other online repositories.

---

## Usage

### Run with an Image

```bash
python age_detection.py -i path/to/image.jpg
```

### Run with a Video File

```bash
python age_detection.py -v path/to/video.mp4
```

### Run with Webcam (Default)

```bash
python age_detection.py
```

### Save Webcam Snapshot

Press `s` during webcam processing to save a snapshot.

### Quit Webcam

Press `q` to exit the video window.

---

## Performance Stats

When running with video:

* Average FPS
* Average Inference Time per Frame

---

## License

This project is intended for educational and non-commercial use only.

---

## Credits

Developed by Jesli

---

## Contact

For queries or collaboration, please contact: \[jeslipriya07@gmail.com](jeslipriya07@gmail.com)
