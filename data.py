

import numpy as np
import os
import glob
import cv2
# %matplotlib widget
import matplotlib.pyplot as plt


def load_data(path, labels=None, grayscale=True, unroll=False, rgb=False, shape=(64, 64)):

    # Get a list of all the folders in the directory
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if labels is not None:
        folders = [f for f in folders if f in labels]

    folders.sort()

    # Print the list of folders
    print("Folders_Found = ", folders)

    data, labels = (), ()

    for folder in folders:
        # Get a list of all the image files in the directory (supported file extensions: .jpg, .jpeg, .png, .bmp, .gif, .tiff)
        images_path = glob.glob(os.path.join(path + "/" + folder, '*.jpg'))


        # By default, the color space of the loaded image is BGR (blue, green, red)
        # rather than the typical RGB (red, green, blue) order.
        images = tuple(cv2.imread(image_path) for image_path in images_path)

        # Original image shape (128, 128) resized to (64, 64) to reduce computational load.
        images = tuple(cv2.resize(image, shape) for image in images)

        # Check to see if we have convert from [bgr or rgb] to [grayscale]
        conversion_state = cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY

        if rgb:
            # Convert images from bgr to rgb from human view
            images = tuple(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images)

        if grayscale:
            # Conversion of image from (BGR/RGB) (128, 128, 3) to greyscale(128, 128)
            images = tuple(cv2.cvtColor(image, conversion_state) for image in images)

        if unroll:
            # Note : No need to unroll now, will do before using this dataset for training
            # Conversion of image from 2d matrix (128, 128) to 1d matrix (128 * 128, )
            images = tuple(image.ravel() for image in images)

        data   += images
        labels += tuple(folder * len(images))


    return np.array(data), np.array(labels)



# Unrolling the 2d matrix (64, 64) to (64 * 64, ) 1d matrix
def unroll_matrix(mat):
    n = mat.shape[0]
    unrolled_mat = tuple(mat[i].ravel() for i in range(n))
    return np.array(unrolled_mat)

# Encoder for encoding labels from ['0' -> 0, 'A' -> 10, 'K' -> 20, 'Z' -> 35]
def encoder(label):
    if label <= '9':
        return ord(label) - 48
    return ord(label) - 55


def encode_labels(labels):
    t = tuple(encoder(label) for label in labels)
    return np.array(t)

# Convert bytes to [KB, MB, GB]
def convert_bytes(size):

    for x in ['bytes', 'KB', 'MB', 'GB']:
        if size < 1024:
            return "%3.1f %s" % (size, x)
        size /= 1024

    return size


# Show a handful of images from the list of training data
# show_random_dataset(images, labels)
def show_random_dataset(image_list, label_list, rowcol=(5, 5)):

    m = image_list.shape[0]

    fig, axes = plt.subplots(rowcol[0], rowcol[1], figsize=(10, 10))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    # widgvis(fig)

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and reshape the image
        # image_random_reshaped = image_list[random_index].reshape((32, 32))

        # Display the image
        ax.imshow(image_list[random_index], cmap='gray', aspect='auto', extent=(20, 80, 20, 80),)

        # Display the label above the image
        ax.set_title(label_list[random_index])
        ax.set_axis_off()

    fig.suptitle("Label, image", fontsize=14)
    plt.show()



# Rotate the image by an angle
def rotate_image(image, angle):
    # Get the image shape and rotation angle

    height, width = image.shape[:2]

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, M, (width, height))

    return rotated_image


def create_dataset_by_rotation(image, label):

    min_angle, max_angle = 10, 30
    sample_size = 200

    left  = np.random.uniform(low= min_angle, high= max_angle, size=(sample_size // 2, ))
    right = np.random.uniform(low=-max_angle, high=-min_angle, size=(sample_size // 2, ))

    sample = np.concatenate((left, right))

    rotated_images = np.array([rotate_image(image, s) for s in sample])
    label_list = [label for _ in range(sample_size)]

    show_random_dataset(rotated_images, label_list)


def create_dataset_by_resize_and_shift(image, label):
    pass
