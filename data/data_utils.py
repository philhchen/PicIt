import functools
import multiprocessing as mp
import ndjson
import os
import random

import cv2
import numpy as np
from pycocotools import mask
import skimage.transform

from .constants import *

def save_training_example(drawing, path, decode=None):
    """
    Saves a single training example to the directory of specified path. The
    filename will be set to the key_id.

    @param drawing - dict: raw data from the Quick! Draw dataset with keys 
                           'word', 'key_id', and 'drawing'
    @param decode (None or "jpg"): whether to decode sketches as images. By
                            default, sketches are saved as ndjson files.
    @param path - str: folder where training examples will be stored.

    @returns str - the filename where the training example is saved.
    """
    ext = '.jpg' if decode == 'jpg' else '.ndjson'
    filename = os.path.join(path, drawing['key_id'] + ext)
    if not os.path.exists(filename):
        drawing_simplified = [{
            'word': drawing['word'],
            'key_id': drawing['key_id'],
            'drawing': drawing['drawing']
        }]
        if decode == 'jpg':
            drawing_decoded = decode_drawing(drawing['drawing'])
            cv2.imwrite(filename, drawing_decoded)
        else:
            with open(filename, mode='w') as f:
                writer = ndjson.dump(drawing_simplified, f)
    result = os.path.join(drawing['word'], drawing['key_id'] + ext)

    # Return only the label with the key_id for sake of space.
    return result

def parse_label(filename, path=RAW_DIR_NAME, decode=None):
    """
    Helper for parse_dataset: parses a single .ndjson file associated with the
    specified path
    @param filename (str): string specifying the path to the .ndjson file to
                            parse
    @param decode (None or "jpg"): whether to decode sketches as images. By
                            default, sketches are saved as ndjson files.
    @param path - str: folder where training examples will be stored.
    """
    list_ids = []
    label, _ = os.path.splitext(filename)

    full_filename = os.path.join(path, filename)
    with open(full_filename) as f:
        if decode == 'jpg':
            dir_name = os.path.join(path, '../img/' + label)
        else:
            dir_name = os.path.join(path, label)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        drawings = ndjson.load(f)
        for drawing in drawings:
            example_filename = save_training_example(drawing, dir_name, decode)
            list_ids.append(example_filename)
    return list_ids

def parse_dataset(path=RAW_DIR_NAME, decode=None, early_return=True):
    """
    Restructures dataset from '.ndjson' files into folders. Each folder will be
    of the form 'dataset/{LABEL}' and will contain 1 file per training example.
    Also saves the list of all filenames to 'filenames.txt'.

    @param path - str: path to directory containing dataset
    @param decode - None or "jpg" - how to decode training examples
    @param early_return - bool: indicates whether method should return early
        if 'filenames.txt' already exists

    @returns list containing all the filenames of the training examples 
        (relative to path)
    @returns list containing all the labels of the dataset
    """
    list_ids = []
    labels = set()

    # If the filenames.txt file already exists, parse the file to find
    # list_ids and labels, and return early
    if decode == 'jpg':
        list_ids_filename = os.path.join(path, '../img/' + 'filenames.txt')
    else:
        list_ids_filename = os.path.join(path, 'filenames.txt')
    if early_return and os.path.exists(list_ids_filename):
        with open(list_ids_filename) as f:
            list_ids = ndjson.load(f)
        for list_id in list_ids:
            label = os.path.basename(os.path.dirname(list_id))
            labels.add(label)
        return list_ids, list(labels)

    # Loop through all '.ndjson' files and split into individual files
    pool = mp.Pool(mp.cpu_count())
    files = os.listdir(path)
    files = [f for f in files if os.path.splitext(f)[1] == '.ndjson']
    list_ids_temp = []

    parse = functools.partial(parse_label, path=path, decode=decode)
    pool.map_async(parse, files, callback=list_ids_temp.extend)
    pool.close()
    pool.join()

    # Convert list_ids_temp from list of lists to just a list
    list_ids = []
    for list_id in list_ids_temp:
        list_ids += list_id

    # Write output to 'dataset/filename.txt' and find all labels
    with open(list_ids_filename, 'w') as f:
        ndjson.dump(list_ids, f)
    for list_id in list_ids:
        label = os.path.basename(os.path.dirname(list_id))
        labels.add(label)
    return list_ids, list(labels)

def decode_drawing(raw_strokes, line_thickness=5, time_color=True,
                   part_color=True, size=224):
    """
    Decodes a drawing from its raw strokes into a numpy array
    @param raw_strokes - list: list containing x, y, and time of each stroke
    @param line_thickness - int: thickness to encode each raw stroke
    @param time_color - bool: whether or not to encode time as color
    @param part_color - bool: whether or not to further encode time as RBG
    @param size - int: number of pixels to resize to

    @returns np.array (size x size x num_channels) - decoded drawing as np array
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for t, stroke in enumerate(raw_strokes):
        part_num = t % 4
        for i in range(len(stroke[0]) - 1):
            color = 255
            if time_color:
                color -= min(t, 10) * 13 + min(i, 40) * 2
            if part_color:
                if part_num == 1:
                    color = (0, color, color)
                elif part_num == 2:
                    color = (color, color, 0)
                elif part_num == 3:
                    color = (color, 0, color)
                else: # if part_num == 0
                    color = (color, color, color)
            p1 = (stroke[0][i], stroke[1][i])
            p2 = (stroke[0][i+1], stroke[1][i+1])
            cv2.line(img, p1, p2, color, line_thickness, cv2.LINE_AA)
    img = skimage.transform.resize(img, (size, size), preserve_range=True)
    return np.array(img, np.uint8)

def affine_transform_drawing(raw_strokes, bounding_box):
    """
    Performs an affine transform of the drawing to fit into the specified
    bounding box.
    @param raw_strokes - list: list containing x, y, and time of each stroke
    @param bounding_box - list: list containing (x1, y1, x2, y2) of the box
                                into which to fit the raw strokes
    @returns transformed_strokes - list: list containing transformed x, y, t
    """
    x1, y1, x2, y2 = bounding_box
    dx, dy = x2 - x1, y2 - y1
    transformed_strokes = []
    for stroke in raw_strokes:
        stroke_x = [(dx * x) // 256 + x1 for x in stroke[0]]
        stroke_y = [(dy * y) // 256 + y1 for y in stroke[1]]
        transformed_strokes.append([stroke_x, stroke_y])
    return transformed_strokes

def get_bounds(raw_strokes):
    """
    Gets the maximum x and y values of the strokes
    @param raw_strokes - list: list containing x, y, and time of each stroke
    @returns max_x, max_y - int, int: the boounds of the strokes
    """
    max_x = max_y = 0
    for stroke in raw_strokes:
        max_x = max(max_x, max(stroke[0]))
        max_y = max(max_y, max(stroke[1]))
    return max_x, max_y

def generate_realistic_sketch(val_image, val_label, val_dataset, quick_draw_dataset, max=5):
    """
    Generates COCO image with QuickDraw images in the same positions. If a label 
    from the COCO image can't be found in the QuickDraw Dataset, it is simply ignored.
    *may result in black images

    @param val_image is a validation image from the COCO Dataset
    @param val_label is the corresponding label from the COCO Dataset
    @param val_dataset is the COCO Dataset module
    @param quick_draw_dataset is the QuickDraw Dataset object
    @param max is the maximum number of objects in the image
    @returns composite sketch, new_boxes, class_ids: the new composite image with corresponding 
    sketches from COCO image, new bounding boxes within the COCO Data frame, and the class_ids of each
    object in the image. 

    """

    bounding_boxes = val_label[:, :4]
    bounding_boxes = bounding_boxes.astype(int)
    class_ids = val_label[:, 4:5]
    quick_size = 256
    coco_size = 224
    X = val_image.shape[0]
    Y = val_image.shape[1]
    draw_indicies = []
    new_boxes = []
    new_classids = []
    for i in range(len(class_ids)):
        if i > max:
            break
        coco_index = (int)(class_ids[i][0])
        label = val_dataset.classes[coco_index]
        if label in quick_draw_dataset.labels_to_images:
            list = quick_draw_dataset.labels_to_images[label]
            rand = np.random.randint(0, len(list))
            draw_img_index = quick_draw_dataset.labels_to_images[label][rand]
            draw_indicies.append(draw_img_index)
            box = bounding_boxes[i]
            resize_box = [(int)(box[0]*quick_size/X), (int)(box[1]*quick_size/Y), (int)(box[2]*quick_size/X), (int)(box[3]*quick_size/Y)]
            new_boxes.append(resize_box)
            new_classids.append(coco_index)

    #new_boxes = np.array(new_boxes)
    #new_classids = np.array(new_classids)
    composite_sketch, segmentations, labels = quick_draw_dataset.generate_composite_drawing(new_boxes, draw_indicies)

    #Resize to COCO frame
    for i, box in enumerate(new_boxes):
        box = [(int)(box[0]*coco_size/quick_size), (int)(box[1]*coco_size/quick_size), (int)(box[2]*coco_size/quick_size), (int)(box[3]*coco_size/quick_size)]
        new_boxes[i] = box
    
    return composite_sketch, segmentations, new_boxes, new_classids

def affine_transform_boxes(boxes, from_scale=256, to_scale=IMG_SIZE):
    scale_fn = lambda x : to_scale * x // from_scale
    return [list(map(scale_fn, box)) for box in boxes]

def get_annotations(boxes, labels, segmentations):
    """
    @returns annotations : list[dict]
    """
    annotations = []
    for box, label, segmentation in zip(boxes, labels, segmentations):
        annotation = {
            'bbox': box,
            'bbox_mode': 0,
            'category_id': label,
            'segmentation': {
                'size': segmentation['size'],
                'counts': segmentation['counts'].decode('utf-8')
            }
        }
        annotations.append(annotation)
    return annotations

def get_segmentation_images(segmentations, labels):
    """
    @param segmentations - list[dict]: list of RLE encoding of segmentation
                binary masks
    @param labels - list: list of indices of labels
    @param coco - QuickDrawDataset

    @returns inst_img
    @returns label_img
    """
    decoded_segmentations = [mask.decode(s) for s in segmentations]
    inst_img = np.zeros(decoded_segmentations[0].shape)
    label_img = np.zeros(decoded_segmentations[0].shape)
    for i, segmentation in enumerate(decoded_segmentations):
        inst_img = np.maximum(inst_img, i * segmentation)
        label_img = np.maximum(label_img, labels[i] * segmentation)
    return inst_img, label_img

if __name__ == '__main__':
    parse_dataset(decode=None)
    parse_dataset(decode='jpg')