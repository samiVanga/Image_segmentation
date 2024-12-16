import cv2
import numpy as np
from matplotlib import pyplot as plt
from pipeline import apply_segmentation
from pipeline import process_images
from postProcessing import postProcessing
from preprocessing import pre
from pipeline import input_image
from evaluation import output
from evaluation import mean_intersection

#this applies the preprocessing function in the Preprocessing.py
#


def watershed_segmentation(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #kernel = np.ones((5, 5), np.uint8)
    #opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    bin_img = cv2.morphologyEx(otsu,
                               cv2.MORPH_OPEN,
                               kernel,
                               iterations=2)
    background = cv2.dilate(bin_img, kernel,iterations=3)

    distTransform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    ret, foreground = cv2.threshold(distTransform, 0.2 * distTransform.max(), 255, cv2.THRESH_BINARY)
    foreground = foreground.astype(np.uint8)

    unknown = cv2.subtract(background, foreground)

    ret, marker = cv2.connectedComponents(foreground)
    marker = marker + 1
    marker[unknown == 255] = 0
    markers = cv2.watershed(img, marker)

    labels = np.unique(markers)

    flower = []
    for label in labels[2:]:
        # Create a binary image in which only the area of the label is in the foreground
        # and the rest of the image is in the background
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        flower.append(contours[0])

    # Draw the outline
    img = cv2.drawContours(img, flower, -1, color=(0, 23, 223), thickness=2)

    return img

if __name__ == "__main__":
    input = process_images("Images", "input-images", input_image)
    gt = process_images("ground_truths", "gt_images", input_image)
    process_images("Images", "preprocessing-images",pre)
    apply_segmentation("Threshold-Output", "watershed-middle-output", watershed_segmentation)
    mask = process_images("watershed-middle-output","watershed-output-final",postProcessing)
    output(input, mask)
    mean_intersection(mask, gt)

