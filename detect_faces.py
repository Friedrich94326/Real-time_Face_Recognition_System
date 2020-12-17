# Import packages for face detection
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
from truncate import *




""" Detect faces using cv2.imread() """

# Detect a single face from a given photo
def MTCNN_detect_face(filename, required_size = (160, 160, 3)):
    
    print('[INFO] Loading image file: ', filename) # for bebugging
    
    img = cv2.imread(filename) # BGR image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converted to RGB format
    
    detector = MTCNN()
    results = detector.detect_faces(img)
   
    n_faces = len(results) # how many faces got detected
        
    # extract a face (the 1st face detected)
    if n_faces > 0:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = img_rgb[y1:y2, x1:x2, :]

        # rsize pixels to the required size for FaceNet model: (160, 160, 3)
        face_image = Image.fromarray(face)  # face_image: PIL.Image.Image
        image = np.resize(face_image, required_size)  # image: ndarray
        face_array = asarray(image)  # face_array: ndarray
        # print(face_array.shape)
        
        # Draw a bounding box on the img
        color = (0, 255, 0)
        img_original = cv2.rectangle(img_original,
                                    (x1, y1),
                                    (x2, y2),
                                    color = color,
                                    thickness = 3
                                    )
        
        return face_array
    else:
        no_face_array = np.zeros(shape = (160, 160, 3))
        return no_face_array

""" Detect faces using PIL.Image.open() """





# Detect a single face from a given photo
def MTCNN_detect_face_2(filename, required_size = (160, 160)):
    
    print('[INFO] Loading image file: ', filename) # for bebugging
    
    img = Image.open(filename) # BGR image
    img = img.convert('RGB') # converted to RGB format
    pixels = asarray(img)
    
    detector = MTCNN()
    results = detector.detect_faces(pixels)
   
    n_faces = len(results) # how many faces got detected
        
    # extract a face (the 1st face detected)
    if n_faces > 0:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]

        # rsize pixels to the required size for FaceNet model: (160, 160, 3)
        img = Image.fromarray(face)  # face_image: PIL.Image.Image
        img = img.resize(required_size)  # image: ndarray
        face_array = asarray(img)  # face_array: ndarray
        # print(face_array.shape)
        
        return face_array
    else:
        no_face_array = np.zeros(shape = (160, 160, 3))
        return no_face_array


""" Load image data to make X & y """

# Load images and extract faces for all images in a directory
def load_faces(dir):
    faces = []
    
    # enumerate files
    for filename in listdir(dir):
        filepath = dir + filename
        face = MTCNN_detect_face_2(filepath) # face: ndarray of shape: (160, 160, 3)
        faces.append(face)
        
    return faces
 
# Load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(dir):
    X = []
    y = []
    
    # enumerate folders, on per class
    for subdir in listdir(dir):
        # path
        filepath = dir + subdir + '/'
        
        # skip any files that might be in the dir
        if not isdir(filepath):
            continue
            
        # load all faces in the subdirectory
        faces = load_faces(filepath)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        
        print('[INFO] Loaded %d examples for class: %s' % (len(faces), subdir))
        
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)




# Load images and extract faces for all images in a directory
def load_faces_2(dir):
    imgs_subdir = []
    
    # enumerate files
    for filename in listdir(dir):
        filepath = dir + filename
        img = MTCNN_detect_face_bounding_box(filepath) # face: ndarray of shape: (160, 160, 3)
        imgs_subdir.append(img)
        
    return imgs_subdir
 
# Load a dataset that contains one subdir for each class that in turn contains images
def load_dataset_2(dir):

    Imgs_DATA_DIR = []
    
    # enumerate folders, on per class
    for subdir in listdir(dir):
        # path
        filepath = dir + subdir + '/'
        
        # skip any files that might be in the dir
        if not isdir(filepath):
            continue
            
        # load all faces in the subdirectory
        imgs_subdir = load_faces_2(filepath)
        
        
        Imgs_DATA_DIR.extend(imgs_subdir)
        
    return asarray(Imgs_DATA_DIR)
