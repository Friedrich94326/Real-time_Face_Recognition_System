from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from numpy import asarray

from solve_cudnn_error import *
solve_cudnn_error()

from truncate import *

detector = MTCNN()
print('MTCNN detector loaded\r\n\r\n\r\n')

required_size = (160, 160) # according to paper

""" image of a single face """

print('image of a single face')
img = cv2.imread(r'ivan.jpg') # Use of an absolute path is advised in an interative window
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = detector.detect_faces(img_rgb) # results: of data type 'list'
print('img detected')

for result in results:
    print(result)

# extract the face


print('\r\n\r\n\r\n')


""" image of multiple faces """

print('image of multiple faces')
img_multiple = cv2.imread(r"yuming.jpg") 
img_rgb_multiple = cv2.cvtColor(img_multiple, cv2.COLOR_BGR2RGB)

results_multiple = detector.detect_faces(img_rgb_multiple)
print('img detected')

for result in results_multiple:
    print(result)

# extract the faces

extract_info = []

n_faces = len(results_multiple) # how many faces got detected

for i in range(n_faces):
    x1, y1, width, height = results_multiple[i]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = img_rgb_multiple[y1:y2, x1:x2]

    prob = results_multiple[i]['confidence']
    prob = truncate(prob, 3) # truncate the confidence to 3 decimals
    prob_str = '(' + str(prob) + ')'

    # draw a rectangle on each face
    color = (0, 255, 0)
    img_rgb_multiple = cv2.rectangle(img_rgb_multiple,
                                    (x1, y1),
                                    (x2, y2),
                                    color = color,
                                    thickness = 3
                                    )

    # write the confidence above the bounding box

    org = ( int( 3 * x1 / 4 + x2 / 4), int(y1 - 20) ) # coordinates of the bottom-left corner of the text string
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = height / 250.
    img_rgb_multiple = cv2.putText(img_rgb_multiple, prob_str, org, font, fontScale, color, 2)


    # rsize pixels to the model size
    image = Image.fromarray(face)
    image = np.resize(image, required_size)
    face_array = asarray(image)

    # store to detect_info
    face_dict = {
        'face_array': face_array,
        'bounding_box': [x1, y1, width, height],
        'prob': prob
        }
    extract_info.append(face_dict)

    

print(extract_info)


# draw bounding box on each face
file_name = 'image with bounding boxes.jpg'
img_rgb_multiple = cv2.cvtColor(img_rgb_multiple, cv2.COLOR_BGR2RGB)
cv2.imshow(file_name, img_rgb_multiple)
key = cv2.waitKey(0)

if key == ord('s'): # 's' for 'save'
    cv2.imwrite(file_name, img_rgb_multiple)

print('\r\n\r\n\r\n')

