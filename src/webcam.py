import cv2 # type: ignore
import time
import argparse
import sys
import os

# Ensure project root is in path so absolute imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import ObjectDetector # type: ignore
from src.utils import draw_detections, get_random_colors, count_objects # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--cam", type=int, default=0, help="Camera Index (default 0)")
    args = parser.parse_args()

    # Initialize Detector
    detector = ObjectDetector(model_path=args.model)
    colors = get_random_colors(len(detector.class_names))

    # Initialize Webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam... Press 'q' to quit.")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detection
        detections, _ = detector.detect(frame, conf_threshold=args.conf)
        
        # Draw
        annotated_frame = draw_detections(frame, detections, colors)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show Counts (Optional overlay)
        counts, total = count_objects(detections)
        y_offset = 60
        for cls, count in counts.items():
            text = f"{cls}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 25

        cv2.imshow("Webcam Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
