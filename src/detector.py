from ultralytics import YOLO
import logging
import os

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLOv8 or YOLO-World model.
        Automatically downloads weights if not present locally.
        """
        self.model_path = model_path
        logging.info(f"Loading model from {model_path}...")

        if "world" in model_path.lower():
            # Only import YOLOWorld if actually needed (avoids CLIP install on standard runs)
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(model_path)
            self.is_world_model = True
        else:
            self.model = YOLO(model_path)
            self.is_world_model = False

        self.class_names = self.model.names

    def set_classes(self, custom_classes):
        """
        Set custom classes for YOLO-World open-vocabulary detection.
        Only supported by YOLO-World models.
        """
        if self.is_world_model:
            self.model.set_classes(custom_classes)
            self.class_names = self.model.names
        else:
            logging.warning("set_classes() is only supported by YOLO-World models.")

    def detect(self, frame, conf_threshold=0.5):
        """
        Run detection on a single frame (numpy array BGR).
        Returns: (list of detection dicts, class_names dict)
        """
        results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names[cls_id]
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'class_id': cls_id,
                'class_name': cls_name,
            })

        return detections, self.class_names
