import os
import random
import time
from data import *
from cvzone.HandTrackingModule import HandDetector


def folder_crop(folder_name):
    # Set the folder name where the images are located
    folder_name = 'archive/ASL_Dataset/Train/A/'

    # Set the path to the Downloads folder
    downloads_path = os.path.expanduser("~/Downloads/")

    # Set the folder path where the images are located
    folder_path = downloads_path + folder_name

    print(folder_path)

    # Set the desired number of images to keep
    num_images_to_keep = 1000

    # Get the list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Check if the number of images is already less than or equal to the desired number
    if len(image_files) <= num_images_to_keep:
        print("The folder already contains {} or fewer images. No deletion required.".format(num_images_to_keep))
    else:
        # Calculate the number of images to delete
        num_images_to_delete = len(image_files) - num_images_to_keep

        # Randomly select images to delete
        images_to_delete = random.sample(image_files, num_images_to_delete)

        # Delete the selected images
        for image in images_to_delete:
            image_path = os.path.join(folder_path, image)
            os.remove(image_path)

        print("Deleted {} images. The folder now contains {} images.".format(num_images_to_delete, num_images_to_keep))



def crop_lines_from_mediapipe_excel_dataset(folder_path):
    folder_path = "./mediapipe_dataset/"  # Replace with the path to the folder containing the CSV files

    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Open the CSV file for reading and writing
            with open(file_path, "r+") as csv_file:
                # Read the contents of the CSV file into a list
                lines = csv_file.readlines()

                # Delete lines after the first 5000
                lines = lines[:4500]

                # Move the file pointer to the beginning of the file
                csv_file.seek(0)

                # Overwrite the contents of the CSV file with the modified lines
                csv_file.writelines(lines)

                # Truncate the file to the new length
                csv_file.truncate()


def cvzone_hand_tracking_module_speedtest():
    # Load data for speedtest
    images, _ = load_data('./Dataset/Indian/', grayscale=False, shape=(128, 128), labels='A')

    # Load the hand detector module from cvzone
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Always fresh load data before running this test
    start = time.time()

    m = images.shape[0] // 10

    for i in range(m):
        _ = detector.findHands(images[i], draw=False)

    end = time.time()

    print(f'Frames to process : {m}')
    print(f'Time taken : {end - start} sec.')
    print(f'Hand detection speed : {m / (end - start)} frames/s.')
