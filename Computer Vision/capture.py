import cv2
import os

# Function to capture images from the webcam
def capture_images(output_folder, max_images):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Counter for captured images
    image_count = 0

    while True:
        ret, frame = cap.read()

        # Display the captured frame
        cv2.imshow('Capture Images', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break
        # If 's' is pressed, save the image
        elif key == ord('s'):
            image_count += 1
            image_path = os.path.join(output_folder, f'image_{image_count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f'Image captured: {image_path}')

            # If the maximum number of images is reached, exit the loop
            if image_count == max_images:
                break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Specify the output folder and the maximum number of images to capture
output_folder = 'captured_images'
max_images = 10

# Capture images from the webcam
capture_images(output_folder, max_images)
