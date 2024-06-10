import unittest
import cv2
from main import detect_license_plate

class TestLicensePlateDetection(unittest.TestCase):
    def test_detection(self):
        # Load the test image
        car = cv2.imread('cars/car17.jpg')

        # Run the detection function
        result_image, detected_text = detect_license_plate(car)

        # Check if any text is detected
        self.assertTrue(detected_text, "No text detected on the license plate")

if __name__ == '__main__':
    unittest.main()
