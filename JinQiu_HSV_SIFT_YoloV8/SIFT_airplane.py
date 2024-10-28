import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the target image
image_path = "video_data/airplane_Image.png"
image = cv2.imread(image_path)

# Convert the image to grayscale for SIFT feature extraction
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT feature extractor
sift = cv2.SIFT_create()

# Extract keypoints and descriptors from the target image
keypoints1, descriptors1 = sift.detectAndCompute(gray_image, None)

# Display the SIFT keypoints on the target image
keypoint_image = cv2.drawKeypoints(
    gray_image, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
plt.imshow(keypoint_image, cmap='gray')
plt.title('SIFT Keypoints on Target Image')
plt.show()
print(f"Number of extracted keypoints: {len(keypoints1)}")

# Load the video and initialize the FLANN matcher
video_path = "video_data/airplane_takeoff.mp4"
cap = cv2.VideoCapture(video_path)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Set the video display window size
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 500

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video has ended or could not read the video frame.")
        break

    # Resize the current frame to 1000x500
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract SIFT features from the current video frame
    keypoints2, descriptors2 = sift.detectAndCompute(gray_frame, None)

    # If descriptors are empty, skip this frame
    if descriptors2 is None or descriptors1 is None:
        print("Not enough keypoints detected in the current frame.")
        continue

    # Match the features between the target image and the current frame
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]

    if len(good_matches) > 10:
        # Extract the matched keypoints' coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the homography matrix and check if it's valid
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = gray_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw a polygon around the detected object in the video frame
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            print("Homography matrix calculation failed.")
    else:
        print("Not enough good matches for object detection.")

    # Display the current frame
    cv2.imshow("Video", frame)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()
