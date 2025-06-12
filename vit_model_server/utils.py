import cv2
import os
import numpy as np
import base64

def annotate_image(image_path, pred_class, bbox_coords):
    img = cv2.imread(image_path)
    
    # Draw bounding box and class name
    x1, y1, x2, y2 = bbox_coords
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, pred_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Convert the annotated image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64