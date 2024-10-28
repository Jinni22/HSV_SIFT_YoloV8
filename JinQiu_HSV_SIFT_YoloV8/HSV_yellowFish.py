import cv2  # Import OpenCV library for image and video processing
import numpy as np  # Import NumPy for array operations
import time  # Import time library to track the elapsed time

print("Starting color segmentation...")  # Confirm that the program has started

# Initialize video capture object to read the video
cap = cv2.VideoCapture("video_data/videoYellowFish/yellowFish.mp4")
if not cap.isOpened():  # Check if the video was opened successfully
    print("Error: Unable to open video file.")
    exit()

print("Video file opened successfully.")  # Confirm that the video file opened correctly

# Create a window to display the trackbars for HSV adjustment
cv2.namedWindow('Trackbars')

# Define a placeholder function for the trackbars
def nothing(x):
    pass

# Create trackbars to control Hue (color), Saturation, and Value (brightness)
cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, nothing)  # Minimum Hue
cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, nothing)  # Maximum Hue
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, nothing)  # Minimum Saturation
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, nothing)  # Maximum Saturation
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, nothing)  # Minimum Value
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, nothing)  # Maximum Value

# List to store the paths of the centroids
paths = []
start_time = time.time()  # Record the start time of the video

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if not ret:
        # If no frame is returned, reset the video to the first frame for looping
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize the frame to reduce its size for faster processing
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the frame from BGR color space to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current values of the trackbars
    hue_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
    hue_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
    sat_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    sat_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    val_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
    val_max = cv2.getTrackbarPos('Val Max', 'Trackbars')

    # Define the lower and upper bounds for the HSV mask
    lower_bound = np.array([hue_min, sat_min, val_min])
    upper_bound = np.array([hue_max, sat_max, val_max])

    # Create a mask based on the HSV bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the frame to draw the tracking results
    output_frame = frame.copy()
    new_centroids = []  # List to store new centroids in the current frame

    # Iterate over all detected contours and draw green bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box coordinates
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

            # Calculate the centroid of the bounding box
            cx = x + w // 2
            cy = y + h // 2
            new_centroids.append((cx, cy))  # Store the centroid coordinates

    # Calculate the elapsed time since the video started
    elapsed_time = time.time() - start_time

    # Only update and draw paths after 100 seconds
    if elapsed_time > 100:
        if len(new_centroids) > 0:
            for centroid in new_centroids:
                found = False
                for path in paths:
                    if len(path) > 0 and np.linalg.norm(np.array(path[-1]) - np.array(centroid)) < 50:
                        # Ensure the centroid is within the green bounding box
                        if x <= centroid[0] <= x + w and y <= centroid[1] <= y + h:
                            path.append(centroid)  # Add the centroid to the path
                            found = True
                            break
                if not found:
                    paths.append([centroid])  # Create a new path for the centroid

        # Keep only the last 25 points in each path
        for path in paths:
            if len(path) > 25:
                del path[:-25]

        # Draw the paths on the output frame
        for path in paths:
            if len(path) > 1:
                for i in range(len(path) - 1):
                    cv2.line(output_frame, path[i], path[i + 1], color=(0, 0, 255), thickness=2)

    # Adjust window size and display the mask and output frame
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Segmented with Bounding Boxes and Path Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 600, 300)
    cv2.resizeWindow('Segmented with Bounding Boxes and Path Tracking', 600, 300)

    cv2.imshow('Mask', mask)  # Display the mask window
    cv2.imshow('Segmented with Bounding Boxes and Path Tracking', output_frame)  # Display the output window

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
