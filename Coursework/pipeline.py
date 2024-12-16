import glob
import os
import cv2
import numpy as np



# this function is to store the preprocessed image in a separate folder
def process_images(input_folder, output_folder, preprocess):
    # this creates the output folder if not already present
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # empty array to add images to
    image_set = []

    # this loops through the input folder
    for subfolder in ['easy', 'medium', 'hard']:
        input_subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)

        # creates the corresponding folders in the processing images folder
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # this applies the preprocessing function to each of the images in the input folder
        for image_path in glob.glob(os.path.join(input_subfolder_path, '*.jpg')):
            # print(f"Processing {image_path}")
            segmented_image = preprocess(image_path)

            segmented_image = np.asarray(segmented_image)  # image needs to be numpy array
            image_set.append(segmented_image)  # add image to image_set

            base_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_subfolder_path, base_filename)
            cv2.imwrite(output_image_path, segmented_image)

        # same preprocessing function but for .png images (which are the format for ground truth images)
        for image_path in glob.glob(os.path.join(input_subfolder_path, '*.png')):
            # print(f"Processing {image_path}")
            segmented_image = preprocess(image_path)

            segmented_image = np.asarray(segmented_image)  # image needs to be numpy array
            image_set.append(segmented_image)  # add image to image_set

            base_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_subfolder_path, base_filename)

            # Save the processed image to the corresponding output subfolder
            cv2.imwrite(output_image_path, segmented_image)

    return image_set

# function to save image as a Python variable
def input_image(image_path):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb



def apply_segmentation(input_folder, output_folder, function):
    # this creates the output folder if not already present
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # this loops through the input folder
    for subfolder in ['easy', 'medium', 'hard']:
        input_subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)

        # creates the corresponding folders in the processing images folder
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # this applies the preprocessing function to each of the images in the input folder
        for image_path in glob.glob(os.path.join(input_subfolder_path, '*.jpg')):
            #print(f"Processing {image_path}")
            segmented_image = function(image_path)

            base_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_subfolder_path, base_filename)

            # Save the processed image to the corresponding output subfolder
            cv2.imwrite(output_image_path, segmented_image)

def apply_split_segmentation(input_folder, output_folder, function):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subfolder in ['easy', 'medium', 'hard']:
        input_subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder_path = os.path.join(output_folder, subfolder)

        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        for image_path in glob.glob(os.path.join(input_subfolder_path, '*.jpg')):
            # Load the image from the path
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            # If you need the image in grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Pass the loaded (and possibly converted) image to the function
            split_merge = function(image=grayscale_image, split_thresh=14, merge_thresh=20)
            masks = split_merge.segmentation()
            canvas = split_merge.visualization(masks=masks)

            base_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_subfolder_path, base_filename)

            # Save the processed image to the corresponding output subfolder
            cv2.imwrite(output_image_path, canvas)

