import cv2
from pipeline import apply_segmentation
import numpy as np


def pre(image_path):
    image = cv2.imread(image_path)  # Read the image from the image path
    bilateral_filtered = cv2.bilateralFilter(image, 80, 200, 200)  # Apply bilateral filtering

    img_hsv = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2HSV)  # converts the RGB image to a HSV image
    _, _, value_channel = cv2.split(img_hsv)  # Extracts the Value channel

    return value_channel

def preprocess(image_path):
    image = cv2.imread(image_path) #reads the image from the image path
    bilateral50d = cv2.bilateralFilter(image, 50, 200, 150) # applies a bilateral filtering to th eimage

    hsv_image = cv2.cvtColor(bilateral50d, cv2.COLOR_BGR2HSV) # converts the RGB image to a HSV image
    _, _, value_channel = cv2.split(hsv_image)  # Extracts the Value channel
    _, otsu_thresh = cv2.threshold(value_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_thresh


def thresh_pre(image_path):
    image = cv2.imread(image_path)  # Read the image from the image path

    bilateral_filtered = cv2.bilateralFilter(image, 100, 100, 200)  # Apply bilateral filtering

    # change brightness and contrast of bilateral_filtered
    brightness = -30
    contrast = 60
    img = np.int16(bilateral_filtered)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    bilateral_filtered = cv2.bilateralFilter(img, 100, 60, 200)  # Another layer of bilateral filtering

    img_hsv = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2HSV)  # converts the RGB image to a HSV image
    _, _, value_channel = cv2.split(img_hsv)  # Extracts the Value channel

    return value_channel



