import os
import json

# Load composite bounding box json file
dir = '../../dataset/composite/'
filename = os.path.join(dir, 'test' + '/data.json')
with open(filename) as f:
    data = json.load(f)

dir = '../../dataset/composite/'
classes = json.load(open(os.path.join(dir, 'train/labels.json')))

def bbox_loss(bounding_boxes, bbox):
    """
    @param bounding_boxes - list: contains list as elements, each specifying a bounding box
    @param bbox - dict: contains keys for one bounding box of Quick! Draw sketch (coordinates, id, etc.)

    Here, loss is defined as (area of overlap) / (total area)

    Returns
        (1) The "loss" of the best match (the maximum is the best)
        (2) The index of the bounding box that best matches it
    """
    # Init
    best_match_loss = -1
    best_match_idx = -1
    overlap = None

    bbox_coords = bbox['bbox']  # Gets bounding box coordinates for the composite image
    for idx in range(len(bounding_boxes)):
        bounding_box = bounding_boxes[idx]
        if (bounding_box[2] > bbox_coords[2]) or (bounding_box[3] > bbox_coords[3]):  # Case with no overlap
            overlap = 0
        else:
            overlap = np.abs(bbox_coords[2] - bounding_box[0]) * np.abs(bbox_coords[3] - bounding_box[1])
        area = np.abs(bbox_coords[2] - bbox_coords[0]) + np.abs(bbox_coords[1] - bbox_coords[3]) 
        area += np.abs(bounding_box[2] - bounding_box[0]) + np.abs(bounding_box[1] - bounding_box[3])
        area -= overlap

        loss = overlap / area

        if loss > best_match_loss:
            best_match_loss = loss
            best_match_idx = idx

    return best_match_loss, best_match_idx


def loss_one_image(bounding_boxes, class_ids, img, class_weight=0.5, bbox_weight=0.5):
    """
    Returns the loss in comparing composite image to one image in the COCO dataset
    @param bounding_boxes - list: contains lists as elements, each specifiying a bounding box 
    for an image
    @param class_ids - list
    @param img - dict: takes in a dictionary of image data
    @param class_weight - float: how much to weight bounding box discrepancy
    @param bbox_weight - float: how much to weight class discrepancy
    """
    annotations = img['annotations']  # List of bbox annotations
    loss = 0
    for bbox in annotations:
        loss_bbox, best_match_idx = bbox_loss(bounding_boxes, bbox) # For the best bbox, loss and best_match_idx
        class_id_best = class_ids[best_match_idx] # Gets label for that box
        class_id_sketch = coco.getCatIds(classes[annotations['category_id']]) # Gets label for the class in COCO
        if class_id_best != class_id_sketch:
            loss_class = 0
        else:
            loss_class = 1
        loss += class_weight * loss_class + bbox_weight * loss_bbox

    return loss

def find_nearest_neighbor(dataset, img):
    best_loss = 0
    best_match = None
    for i in range(len(dataset)):
        val_image, val_label = dataset[i]  # Display random image
        bounding_boxes = val_label[:, :4]
        class_ids = val_label[:, 4:5]
        loss = loss_one_image(bounding_boxes, class_ids, img)

        if loss > best_loss:
            best_loss = loss
            best_match = val_image

    return best_match