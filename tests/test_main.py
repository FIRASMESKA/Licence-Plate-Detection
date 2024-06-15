import unittest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import process_image  # Now the import should work

class TestLicensePlateRecognition(unittest.TestCase):
    def test_process_image(self):
        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cars/car17.jpg'))
        car, detection, plate_cnt, plate_thresh = process_image(image_path)

        # Check that the car image is not None
        self.assertIsNotNone(car, "The car image should not be None.")

        # Check that the plate contour is not None
        self.assertIsNotNone(plate_cnt, "The plate contour should not be None.")

        # Check that the thresholded plate image is not None
        self.assertIsNotNone(plate_thresh, "The thresholded plate image should not be None.")

        # Check that detection contains at least one item
        self.assertGreater(len(detection), 0, "Detection list should contain at least one item.")

        # Check that the first detection contains text and confidence fields
        detected_text, confidence = detection[0][1], detection[0][2]
        self.assertIsInstance(detected_text, str, "Detected text should be a string.")
        self.assertIsInstance(confidence, float, "Confidence should be a float.")
        self.assertGreaterEqual(confidence, 0, "Confidence should be non-negative.")
        self.assertLessEqual(confidence, 1, "Confidence should be less than or equal to 1.")

if __name__ == '__main__':
    unittest.main()
