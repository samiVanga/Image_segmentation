import cv2
import numpy as np

from preprocessing import pre
from Threshold import threshold
from pipeline import process_images
from preprocessing import thresh_pre
from pipeline import input_image
from pipeline import apply_segmentation
from postProcessing import postProcessing
from evaluation import output
from evaluation import mean_intersection



def canny(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    canny_edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)

    # Perform closing operation to close gaps in edges
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

    # Perform dilation to thicken the borders
    #thickened_edges = cv2.dilate(closed_edges, kernel,
                                # iterations=3)  # Increase the number of iterations for thicker borders

    # Perform closing operation to fill in large gaps in the border
    large_gaps_filled = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(large_gaps_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill in contours
    filled_image = np.zeros_like(img)
    cv2.drawContours(filled_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)


    return filled_image
if __name__ == "__main__":
    gt = process_images("ground_truths", "gt_images", input_image)
    i_image = process_images("Images", "input-images", input_image)
    process_images("Images", "preprocessing-images", pre)
    apply_segmentation("preprocessing-images", "canny_middle_output",threshold)
    mask = process_images("canny_middle_output","canny_output_final",canny)
    output(i_image, mask)
    mean_intersection(mask, gt)
