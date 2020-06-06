import json
import multiprocessing as mp
import ndjson
import os
import random

import numpy as np
from pycocotools import mask
from torch.utils import data

from .constants import *
from .data_utils import *

class QuickDrawDataset(data.Dataset):
    """
    A data loader where the images are arranged in this way:
    root/dog/xxx.ndjson
    root/dog/xxy.ndjson

    root/cat/123.ndjson
    root/cat/nsdf3.ndjson
    """
    def __init__(self, root=RAW_DIR_NAME, img_size=IMG_SIZE):
        """
        @param root - str: Root directory path.
        @param img_size - int: the size to convert images in the dataset
        """
        self.list_IDs, self.labels = parse_dataset(path=root, decode=None)
        self.root = root
        self.img_size = img_size
        self.labels_to_indices = {}

        for i, label in enumerate(self.labels):
            self.labels_to_indices[label] = i

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index, decode=True):
        """
        @returns drawing_decoded - np.array (img_size x img_size x 3): drawing 
                                    decoded as a np array
        @returns index - index of the correct label associated with the drawing
        """
        filename = os.path.join(self.root, self.list_IDs[index])
        with open(filename) as f:
            drawing = ndjson.load(f)
        drawing_decoded = drawing[0]['drawing']
        if decode:
            drawing_decoded = decode_drawing(drawing[0]['drawing'],
                                             size=self.img_size)
        label = self.labels_to_indices[drawing[0]['word']]
        return drawing_decoded, label
    
    def get_binary_segmentation(self, drawing):
        """
        @param drawing - list containing x, y, and time of each stroke of
                         the drawing
        @returns dict - per-pixel segmentation mask 
        """
        img_mask = decode_drawing(drawing).sum(axis=-1) != 0

        min_nonzero = np.where(img_mask.any(axis=0), img_mask.argmax(axis=0), -1)
        val = img_mask.shape[0] - np.flip(img_mask, axis=0).argmax(axis=0) - 1
        max_nonzero = np.where(img_mask.any(axis=0), val, -1)

        bimask_x = np.zeros(img_mask.shape, dtype='uint8')
        for i in range(len(bimask_x)):
            if min_nonzero[i] != -1:
                bimask_x[min_nonzero[i] : max_nonzero[i] + 1, i] = True
        
        min_nonzero = np.where(img_mask.any(axis=1), img_mask.argmax(axis=1), -1)
        val = img_mask.shape[1] - np.flip(img_mask, axis=1).argmax(axis=1) - 1
        max_nonzero = np.where(img_mask.any(axis=1), val, -1)

        bimask_y = np.zeros(img_mask.shape, dtype='uint8')
        for i in range(len(bimask_y)):
            if min_nonzero[i] != -1:
                bimask_y[i, min_nonzero[i] : max_nonzero[i] + 1] = True

        bimask = np.logical_and(bimask_x, bimask_y)
        bimask_encoded = mask.encode(np.asarray(bimask, order="F"))
        return bimask_encoded

    def get_random_composite_drawing(self, num_components):
        """
        Generates a random composite drawing by creating bounding boxes
        for a randomly selected set of drawings and creating the composite
        sketch formed by the drawings
        @return composite_sketch, boxes, labels
        """
        # Helper to ensure boxes are sufficiently far apart
        def is_valid_box(boxes, new_box, slack=0):
            x1, y1, x2, y2 = new_box
            for box in boxes:
                X1, Y1, X2, Y2 = box
                if not (X1 > x2 - slack or Y1 > y2 - slack
                        or x1 > X2 - slack or y1 > Y2 - slack):
                    return False
            return True

        # Helper to get in-bounds anchor point for a given scale
        def get_anchor_point(scale, bounds):
            x1 = random.randint(0, (int)(255 - scale * bounds[0]))
            y1 = random.randint(0, (int)(255 - scale * bounds[1]))
            return x1, y1
        
        def get_boxes(x1, y1, scale, bounds):
            x2_f, y2_f = (int)(x1 + scale * 256), (int)(y1 + scale * 256)
            x2_b, y2_b = (int)(x1 + scale * bounds[0]), (int)(y1 + scale * bounds[1])
            frame_box = [x1, y1, x2_f, y2_f]
            bounding_box = [x1, y1, x2_b, y2_b]
            return frame_box, bounding_box

        # Get random indices for which drawings to choose from
        indices = np.random.choice(self.__len__(), num_components)

        # Get random factors to resize each drawing
        scale_mean = np.sqrt(0.4 / num_components)
        scale_stdev = np.sqrt(0.05 / num_components)
        scales = np.random.normal(scale_mean, scale_stdev,
                                  num_components).clip(0.15, 0.85)

        # Slack to determine how much images can overlap (number of pixels)
        slack = (int)(np.sqrt(500 / num_components))

        # Generate bounding boxes using the scale factors
        frame_boxes = []
        bounding_boxes = []
        for i in range(num_components):
            drawing, label = self.__getitem__(indices[i], False)
            bounds = get_bounds(drawing)
            scale = scales[i]
            x1, y1 = get_anchor_point(scale, bounds)
            frame_box, bounding_box = get_boxes(x1, y1, scale, bounds)

            while not is_valid_box(bounding_boxes, bounding_box, slack):
                scale *= 0.99
                x1, y1 = get_anchor_point(scale, bounds)
                frame_box, bounding_box = get_boxes(x1, y1, scale, bounds)
            frame_boxes.append(frame_box)
            bounding_boxes.append(bounding_box)

        composite_sketch, segmentations, labels = self.generate_composite_drawing(frame_boxes, indices)
        return composite_sketch, bounding_boxes, segmentations, labels

    def generate_composite_drawing(self, boxes, indices):
        """
        Helper method for 
        Generates a composite drawing given the bounding boxes and labels of 
        the individual components
        @param boxes - ndarray (N x 4): the predicted boxes in [x1, y1, x2, y2]
        @param indices - list - int (len N): the indices of the labels to add

        @returns list: list containing x, y, and time of each stroke of
                        the composite drawing
        @returns list (int): labels
        """
        assert len(boxes) == len(indices)
        composite_sketch = []
        segmentations = []
        labels = []
        for i, box in enumerate(boxes):
            raw_strokes, label = self.__getitem__(indices[i], False)
            transformed = affine_transform_drawing(raw_strokes, box)
            composite_sketch += transformed
            segmentations.append(self.get_binary_segmentation(transformed))
            labels.append(label)
        return composite_sketch, segmentations, labels

def create_composite_dataset(count, mode, quickdraw_dataset,
                             root_composite=COMPOSITE_DIR_NAME,
                             min=2, max=8):
    """
    Generates a composite dataset from concatenating raw sketches.
    @param count - int: number of images to generate
    @param mode - str ('train', 'val', or 'test'): mode for composite dataset
    @param root_raw - str: path to directory containing folders of raw sketches
    @param root_composite - str: path to directory to save composite dataset
    @param min - int: minimum number of raw images per composite image (inclusive)
    @param max - int: maximum number of raw images per composite image (inclusive)
    """
    if mode not in ['train', 'val', 'test']:
        print('Error: mode must be train, val, or test.')
        return
    if not os.path.exists(root_composite):
        os.makedirs(root_composite)
    dir_name = os.path.join(root_composite, mode)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if quickdraw_dataset == None:
        quickdraw_dataset = QuickDrawDataset(root=RAW_DIR_NAME)
    nums = np.random.randint(min, max + 1, count)

    # Get the specified number of random composite images
    img_infos = []
    for i, num in enumerate(nums):
        composite_sketch, boxes, segmentations, labels = quickdraw_dataset.get_random_composite_drawing(num)
        img = decode_drawing(composite_sketch)
        boxes = affine_transform_boxes(boxes)
        filename_img = os.path.join(dir_name, '{:0>7d}.jpg'.format(i))
        cv2.imwrite(filename_img, img)
        img_infos.append({
            'file_name': filename_img,
            'height': IMG_SIZE,
            'width': IMG_SIZE,
            'image_id': i,
            'annotations': get_annotations(boxes, labels, segmentations)
        })

    # Write image infos to json file
    filename_data = os.path.join(dir_name, 'data.json')
    filename_labels = os.path.join(dir_name, 'labels.json')
    with open(filename_data, 'w') as f:
        json.dump(img_infos, f)
    with open(filename_labels, 'w') as f:
        labels_dict = quickdraw_dataset.labels_to_indices
        labels = ['' for i in range(len(labels_dict))]
        for label in labels_dict:
            labels[labels_dict[label]] = label
        json.dump(labels, f)
    