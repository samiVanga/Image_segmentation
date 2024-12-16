import cv2


def postProcessing(image_path):
    image=cv2.imread(image_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # Perform erosion followed by dilation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opened_image