import cv2
import os
from xml.etree import ElementTree

def annotate_image(image_path, pred_class):
    ann_path = image_path.replace("images", "annotations").replace(".jpg", ".xml")
    img = cv2.imread(image_path)
    if not os.path.exists(ann_path):
        return image_path

    tree = ElementTree.parse(ann_path)
    root = tree.getroot()

    for obj in root.iter("object"):
        cls_name = obj.find("name").text
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = map(int, [box.find(t).text for t in ("xmin", "ymin", "xmax", "ymax")])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, cls_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    save_path = image_path.replace(".jpg", f"_annotated.jpg")
    cv2.imwrite(save_path, img)
    return save_path