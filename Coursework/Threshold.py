import cv2
import numpy as np
from pipeline import process_images
from preprocessing import thresh_pre
from pipeline import input_image
from pipeline import apply_segmentation
from postProcessing import postProcessing
from evaluation import output
from evaluation import mean_intersection

def threshold(image_path):
    image = cv2.imread(image_path)

    # Apply Otsu thresholding to obtain the binary mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_thresh

if __name__ == "__main__":
    gt = process_images("ground_truths", "gt_images", input_image)
    i_image = process_images("Images", "input-images", input_image)
    process_images("Images","preprocessing-images", thresh_pre)
    apply_segmentation("preprocessing-images", "threshold-middle-output", threshold)
    mask = process_images("threshold-middle-output", "Threshold-Output", postProcessing)
    output(i_image, mask)
    mean_intersection(mask, gt)

