# model_live_check.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

torch.cuda.empty_cache()
# Use relative path to the models directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
model = YOLO(model_path)

def detect_and_segment(image, confidence_threshold=0.1, iou_threshold=0.5):
    """
    Detect and segment objects in the image.
    
    Args:
        image: Input image
        confidence_threshold: Minimum confidence for detection (lower for training data)
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detections with masks, bounding boxes, and class IDs
    """
    # Predict with lower confidence threshold for training dataset images
    results = model.predict(
        source=image, 
        show=False, 
        save=False, 
        stream=False,
        conf=confidence_threshold,  # Lower confidence threshold
        iou=iou_threshold,         # IoU threshold for NMS
        verbose=False
    )

    detections = []
    for result in results:
        if result.masks is not None and len(result.masks) > 0:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy()):
                # Get confidence score
                confidence = result.boxes.conf[i].cpu().numpy()
                
                # Accept all detections for training dataset images
                # (since these are images the model was trained on)
                mask = result.masks.data[i].cpu().numpy()
                box = result.boxes.xyxy[i].cpu().numpy()
                
                detections.append({
                    'mask': mask,
                    'bbox': box,
                    'class_id': int(cls_id),
                    'confidence': float(confidence)
                })
                
                print(f"Detected object {i+1}: Class {int(cls_id)}, Confidence: {confidence:.3f}")
    
    if not detections:
        print("No objects detected. Trying with even lower confidence threshold...")
        # Try with even lower confidence if no detections
        results = model.predict(
            source=image, 
            show=False, 
            save=False, 
            stream=False,
            conf=0.05,  # Very low confidence threshold
            iou=0.3,    # Lower IoU threshold
            verbose=False
        )
        
        for result in results:
            if result.masks is not None and len(result.masks) > 0:
                for i, cls_id in enumerate(result.boxes.cls.cpu().numpy()):
                    confidence = result.boxes.conf[i].cpu().numpy()
                    mask = result.masks.data[i].cpu().numpy()
                    box = result.boxes.xyxy[i].cpu().numpy()
                    
                    detections.append({
                        'mask': mask,
                        'bbox': box,
                        'class_id': int(cls_id),
                        'confidence': float(confidence)
                    })
                    
                    print(f"Detected object {i+1}: Class {int(cls_id)}, Confidence: {confidence:.3f}")
    
    print(f"Total detections: {len(detections)}")
    return detections
