# License Plate Recognizer
This project demonstrates how to detect and read text from a vehicle's license plate using OpenCV for image processing and EasyOCR for text recognition.
## Features

- Detect vehicle license plates in images
- Extract text from license plates using EasyOCR
- Display results with confidence scores
- Simple GUI for displaying images using tkinter

## Requirements

- Python 3.x
- OpenCV
- EasyOCR
- tkinter
- PIL (Pillow)

## Installation

Install the necessary libraries using pip:

```bash
pip install opencv-python easyocr pillow
```

## Usage
1. Load and Process Image:

- The script loads an image of a car, resizes it, converts it to grayscale, applies a bilateral filter to preserve edges while reducing noise, and detects edges using the Canny edge detector.
- It then finds contours in the image, sorts them by area, and approximates the largest contours to polygons to identify the license plate.

2. Text Recognition:

- The identified license plate region is extracted and thresholded to improve OCR accuracy.
- EasyOCR is used to read the text from the license plate.

3. Display Results:

- The script displays the detected text on the original image.
- If the text is detected successfully, the license plate area is highlighted and the text along with its confidence score is displayed.
- If no text is detected, an appropriate message is displayed on the image.

4. Custom GUI:

- The final image with the detected license plate is displayed in a tkinter window.
# Notes
  - Ensure that the input image is clear and the license plate is visible for better text detection accuracy.
  - You can modify the script to adjust the preprocessing steps (e.g., different filters or edge detection parameters) to suit different images.
  - 
"This README was developed by [Firas Meskaoui]. If you have any questions or suggestions, feel free to contact me!"
