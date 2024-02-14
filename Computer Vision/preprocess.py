import cv2
import os

# Function to preprocess images
def preprocess_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the image (optional)
            resized_image = cv2.resize(gray_image, (100, 100))  # Adjust size as needed

            # Save the preprocessed image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)

            print(f'Image preprocessed: {output_path}')

# Specify the input and output folders
input_folder = 'captured_images'
output_folder = 'preprocessed_images'

# Preprocess the images
preprocess_images(input_folder, output_folder)
