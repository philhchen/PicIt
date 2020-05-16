import os
import ndjson
import random

import numpy as np
from torch.utils import data

from .data_utils import *

DIR_NAME = os.path.join(
    os.path.dirname(__file__), '../dataset/raw'
)

class QuickDrawDataset(data.Dataset):
    """
    A data loader where the images are arranged in this way:
    root/dog/xxx.ndjson
    root/dog/xxy.ndjson

    root/cat/123.ndjson
    root/cat/nsdf3.ndjson
    """
    def __init__(self, root=DIR_NAME, img_size=224):
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

    def get_random_composite_drawing(self, num_components):
        """
        Generates a random composite drawing by creating bounding boxes
        for a randomly selected set of drawings and creating the composite
        sketch formed by the drawings
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

        # Get random indices for which drawings to choose from
        indices = np.random.choice(self.__len__(), num_components)

        # Get random factors to resize each drawing
        scale_mean = np.sqrt(0.4 / num_components)
        scale_stdev = np.sqrt(0.05 / num_components)
        scales = np.random.normal(scale_mean, scale_stdev,
                                  num_components).clip(0.15, 0.85)

        # Slack to determine how much images can overlap (number of pixels)
        slack = (int)(np.sqrt(2000 / num_components))

        # Generate bounding boxes using the scale factors
        boxes = []
        for i in range(num_components):
            drawing, label = self.__getitem__(indices[i], False)
            bounds = get_bounds(drawing)
            scale = scales[i]
            x1, y1 = get_anchor_point(scale, bounds)
            x2, y2 = (int)(x1 + scale * 256), (int)(y1 + scale * 256)
            box = [x1, y1, x2, y2]

            while not is_valid_box(boxes, box, slack):
                scale *= 0.99
                x1, y1 = get_anchor_point(scale, bounds)
                x2, y2 = (int)(x1 + scale * 256), (int)(y1 + scale * 256)
                box = [x1, y1, x2, y2]
            boxes.append(box)

        composite_sketch, labels = self.generate_composite_drawing(boxes, indices)
        return composite_sketch, labels

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
        labels = []
        for i, box in enumerate(boxes):
            raw_strokes, label = self.__getitem__(indices[i], False)
            transformed = affine_transform_drawing(raw_strokes, box)
            composite_sketch += transformed
            labels.append(label)
        return composite_sketch, labels