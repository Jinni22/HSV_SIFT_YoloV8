import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (based on the COCO dataset)
model = YOLO('yolov8n.pt')  # Use the lightweight 'yolov8n.pt' model

# Define the path to the car video
cars_video_path = "video_data/cars_flow/cars_flow.mp4"

def process_cars_video(video_path, window_name="Car Detection"):
    """Process the car video and display detection results."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to read frame.")
            break

        # Resize the frame to 1200x600 for consistent display
        frame = cv2.resize(frame, (1200, 600))

        # Use the YOLOv8 model to detect objects in the current frame
        results = model(frame)

        # Filter detections to only show vehicles (car, bus, truck)
        vehicle_classes = [2, 5, 7]  # Vehicle classes in the COCO dataset
        vehicle_results = [
            det for det in results[0].boxes.data if int(det[5]) in vehicle_classes
        ]

        # Annotate the frame with vehicle detection results
        annotated_frame = frame.copy()
        for det in vehicle_results:
            x1, y1, x2, y2, conf, cls = det
            label = f'{model.names[int(cls)]} {conf:.2f}'
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

# Process the car video
print("Processing Cars Video...")
process_cars_video(cars_video_path)
