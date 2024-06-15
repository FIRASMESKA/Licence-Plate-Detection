import cv2
from easyocr import Reader
import tkinter as tk
from PIL import Image, ImageTk


def process_image(image_path):
    # Load the image
    car = cv2.imread(image_path)
    if car is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")

    car = cv2.resize(car, (800, 600))  # Resize the image for better visibility
    gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detect edges using the Canny edge detector
    edged = cv2.Canny(filtered, 30, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Sort contours by area and take the top 10

    # Find the license plate
    plate_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)  # Calculate the arc length of the contour
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)  # Approximate the contour to a polygon
        if len(approx) == 4:  # If the polygon has 4 vertices, it's likely to be the license plate
            plate_cnt = approx
            break

    if plate_cnt is None:
        return car, None, None, None

    (x, y, w, h) = cv2.boundingRect(plate_cnt)  # Get the bounding box of the license plate
    plate = gray[y:y + h, x:x + w]  # Extract the license plate from the grayscale image

    # Apply threshold to improve OCR accuracy
    _, plate_thresh = cv2.threshold(plate, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use EasyOCR to read the text from the license plate
    reader = Reader(['en', 'ar'], gpu=False, verbose=False)
    detection = reader.readtext(plate_thresh)
    return car, detection, plate_cnt, plate_thresh


def show_image(image):
    root = tk.Tk()

    # Convert the OpenCV image to a PIL image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    label = tk.Label(root, image=image)
    label.pack()

    root.mainloop()


def annotate_image(car, detection, plate_cnt, plate_thresh):
    if detection:
        cv2.drawContours(car, [plate_cnt], -1, (0, 255, 0), 3)
        text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
        cv2.putText(car, text, (plate_cnt[0][0][0], plate_cnt[0][0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), 2)
        print(text)
        cv2.imshow('license plate', plate_thresh)
    else:
        text = "Impossible to read the text from the license plate"
        cv2.putText(car, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)

    # Display the image
    show_image(car)


if __name__ == "__main__":
    image_path = 'cars/car17.jpg'
    car, detection, plate_cnt, plate_thresh = process_image(image_path)
    annotate_image(car, detection, plate_cnt, plate_thresh)
