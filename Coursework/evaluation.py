import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np

# function to observe segmentation of flower from background
def output(input_images, mask_images):
    segmented_images = []  # List to store segmented images

    for input_image, mask_image in zip(input_images, mask_images):
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        segmented_image = cv2.bitwise_and(input_image, input_image, mask=mask_gray)
        segmented_images.append(segmented_image)  # Append segmented image to the list

    return segmented_images



# intersection over ground truth
def intersection_over_gt(output, gt):
    intersection = cv2.bitwise_and(gt, output)
    return cv2.countNonZero(intersection) / cv2.countNonZero(gt)

def mean_intersection(output, gt):

    mean_intersection_positive = []
    mean_intersection_negative = []

    for x, y in zip(output, gt):
        # make sure that binary image is grayscale
        output_gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        # convert ground truth to grayscale
        gt_gray = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)

        # separate ground truth into foreground and background
        gt_negative = cv2.inRange(gt_gray, 100, 125)
        gt_positive = cv2.inRange(gt_gray, 25, 50)

        # perform operation
        positive = intersection_over_gt(gt_positive, output_gray)
        mean_intersection_positive.append(positive)
        negative = intersection_over_gt(gt_negative, cv2.bitwise_not(output_gray))
        mean_intersection_negative.append(negative)


    folders=["easy","medium","hard"]
    numbers= ["1","2","3"]
    total =0.0

    for folder in folders:
        for number in numbers:
            index = folders.index(folder) * len(numbers) + numbers.index(number)
            positive_ratio = mean_intersection_positive[index]
            #negative_ratio = mean_intersection_negative[index]
            print(f"{folder} {number} - Positive: {positive_ratio}")


    for i in mean_intersection_positive:
        total += i


    overall_performance = total / len(mean_intersection_positive)
    print(f"Overall performance: {overall_performance}")






