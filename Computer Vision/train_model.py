import cv2
import os
import numpy as np

def train_model(input_folder):
    # Create lists to store faces and corresponding labels
    faces = []
    labels = []

    # Get the list of image filenames in the input folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Loop through each image file
    for image_path in image_files:
        # Read the image and convert it to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract the label (person's identity) from the filename
        label = os.path.splitext(os.path.basename(image_path))[0].split('_')[-1]

        # Append the face and label to the lists
        faces.append(image)
        labels.append(int(label))  # Convert label to integer

    # Convert the lists to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)

    # Create the LBPH (Local Binary Patterns Histograms) face recognizer model
    model = cv2.face_LBPHFaceRecognizer.create()

    # Train the model
    model.train(faces, labels)

    # Save the trained model to a file
    model.save('face_recognition_model.yml')

    print('Model trained successfully!')

# Specify the input folder containing preprocessed images
input_folder = 'preprocessed_images'

# Train the facial recognition model
train_model(input_folder)
