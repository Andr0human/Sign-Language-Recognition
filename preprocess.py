
# File for preprocessing images before using them for model training.

'''
    Hand Recognition
    image preprocessing
        Image Resizing
        Segmentation
            BGR to HSV
            Masked
            Canny Edge
        Feature Extraction
'''



import cv2
import numpy as np
import matplotlib.pyplot as plt


# Preprocess the individual frame before using it for classification
def apply_preprocessing(frame):
    # Assumes the frame to be preprocessed is in bgr-format

    # Define the lower and upper bounds of the skin color in HSV color space
    skin_lower = np.array([0, 20, 70], dtype=np.uint8)
    skin_upper = np.array([20, 255, 255], dtype=np.uint8)

    # Convert the frame from BGR color space to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to filter out the skin color
    mask = cv2.inRange(hsv, skin_lower, skin_upper)

    # Apply the mask to the original frame
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    image = cv2.cvtColor(skin, cv2.COLOR_HSV2RGB)
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray


# Preprocessing on the list of images
def get_preprocessed_set(images, image_preprocessor):

    m = images.shape[0]
    preprocessed_images = tuple(image_preprocessor(images[i]) for i in range(m))
    return np.array(preprocessed_images)


# Skin segmentation and contour-based hand detection
def apply_contour_preprocessing(frame):

    # Define the lower and upper bounds of the skin color in HSV color space
    skin_lower = np.array([0, 20, 70], dtype=np.uint8)
    skin_upper = np.array([20, 255, 255], dtype=np.uint8)

    # Define the kernel size for the morphological operations
    kernel_size = 5

    # Convert the frame from BGR color space to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to filter out the skin color
    mask = cv2.inRange(hsv, skin_lower, skin_upper)

    # Apply morphological operations to remove noise and smooth the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a bounding box around the largest contour (assumed to be the hand)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        mask = np.zeros_like(frame)
        mask[y:y+h, x:x+w] = 255
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, (64, 64))
    return frame



# Show frame before and after preprocessing 
def show_preprocess_result(frame, p_frame):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    # widgvis(fig)

    # Display the first image in the left subplot
    axes[0].imshow(frame)
    axes[0].set_title('Frame')

    # Display the second image in the right subplot
    axes[1].imshow(p_frame)
    axes[1].set_title('Processed Frame')

    # Show the plot
    plt.show()

