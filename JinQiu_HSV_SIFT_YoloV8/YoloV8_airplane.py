import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a lightweight model

# Define the path for the airplane video
airplane_video_path = "video_data/airplane_takeoff.mp4"

def process_airplane_video(video_path, window_name="Airplane Detection"):
    """Process the airplane video and display detections."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to read frame.")
            break

        # Resize the frame to 1200x600 for consistent display
        frame = cv2.resize(frame, (1200, 600))

        # Use the YOLOv8 model to detect objects in the current frame
        results = model(frame)

        # Filter detections to only show airplanes (class id = 4 in COCO)
        airplane_results = [
            det for det in results[0].boxes.data if int(det[5]) == 4
        ]

        # Annotate the frame with only airplane detections
        annotated_frame = frame.copy()
        for det in airplane_results:
            x1, y1, x2, y2, conf, cls = det
            label = f'Airplane {conf:.2f}'
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)

        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Process the airplane video
print("Processing Airplane Video...")
process_airplane_video(airplane_video_path)
