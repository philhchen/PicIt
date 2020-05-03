import cv2
import ndjson
import numpy as np
import os
import random
import skimage.transform

DIR_NAME = os.path.join(
    os.path.dirname(__file__), '../dataset'
)

def save_training_example(drawing, path, list_ids=None):
    """
    Saves a single training example to the directory of specified path. The
    filename will be set to the key_id. If list_ids is not None, the filename
    will be added to list_ids.

    @param drawing - dict: raw data from the Quick! Draw dataset with keys 
                           'word', 'key_id', and 'drawing'
    @param path - str: folder where training examples will be stored
    @param list_ids - list: list of all the filenames of the training examples
    """
    filename = os.path.join(path, drawing['key_id'] + '.ndjson')
    if list_ids != None:
        list_ids.append(filename)
    if os.path.exists(filename):
        return

    drawing_simplified = [{
        'word': drawing['word'],
        'key_id': drawing['key_id'],
        'drawing': drawing['drawing']
    }]

    with open(filename, mode='w') as f:
        writer = ndjson.dump(drawing_simplified, f)

def parse_dataset(path=DIR_NAME):
    """
    Restructures dataset from '.ndjson' files into folders. Each folder will be
    of the form 'dataset/{LABEL}' and will contain 1 file per training example.
    Also saves the list of all filenames to 'filenames.txt'.

    @param path - str: path to directory containing dataset
    @returns list containing all the filenames of the training examples 
        (relative to path)
    @returns list containing all the labels of the dataset
    """
    directory = os.fsencode(path)
    list_ids = []
    labels = set()
    
    # If the filenames.txt file already exists, parse the file to find
    # list_ids and labels, and return early
    list_ids_filename = os.path.join(path, 'filenames.txt')
    if os.path.exists(list_ids_filename):
        with open(list_ids_filename) as f:
            list_ids = ndjson.load(f)
        for list_id in list_ids:
            label = os.path.basename(os.path.dirname(list_id))
            labels.add(label)
        return list_ids, labels

    # Loop through all '.ndjson' files and split into individual files
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        label, ext = os.path.splitext(filename)
        labels.add(label)
        if ext != '.ndjson':
            continue

        full_filename = os.path.join(path, filename)
        with open(full_filename) as f:
            dir_name = os.path.join(path, label)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            drawings = ndjson.load(f)
            for drawing in drawings:
                save_training_example(drawing, dir_name, list_ids)

    with open(list_ids_filename, 'w') as f:
        ndjson.dump(list_ids, f)
    return list_ids, list(labels)

def decode_drawing(raw_strokes, line_thickness=5, time_color=True,
                   part_color=True, num_channels=3, size=128):
    """
    Decodes a drawing from its raw strokes into a numpy array
    @param raw_strokes - list: list containing x, y, and time of each stroke
    @param line_thickness - int: thickness to encode each raw stroke
    @param time_color - bool: whether or not to encode time as color
    @param part_color - bool: whether or not to further encode time as RBG
    @param num_channels - int: number of color channels
    @param size - int: number of pixels to resize to

    @returns np.array (256 x 256 x num_channels) - decoded drawing as np array
    """
    img = np.zeros((256, 256, num_channels), dtype=np.uint8)
    for t, stroke in enumerate(raw_strokes):
        part_num = int(float(t) / len(raw_strokes) * num_channels)
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 20) * 10 if time_color else 255
            if part_color:
                if part_num == 1:
                    color = (0, color, color)
                elif part_num == 2:
                    color = (0, 0, color)
                else: # if part_num == 0:
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

if __name__ == '__main__':
    parse_dataset()