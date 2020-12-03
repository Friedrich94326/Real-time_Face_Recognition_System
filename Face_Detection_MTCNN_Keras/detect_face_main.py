from __future__ import print_function
from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from numpy import asarray
import time
import tensorflow as tf
from solve_cudnn_error import *
import imutils # to fasten FPS
from truncate import *


from imutils.video import WebcamVideoStream  # another approach to get images from our webcam (rather than cv2.videoCapture()) 
from imutils.video import FPS
import argparse


def video_init(is_2_write = False, save_path = None):
   
    cap = cv2.VideoCapture(1) # video input selection: 0 for built-in camera, 1 for Logitech C170


    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # C170 default: 768, built-in webcam default: 480
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # C170 default: 1024, built-in webcam default: 640

    # adjust frame size
    frame_width = 512
    frame_height = 384
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

       
    if is_2_write is True:
        
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(frame_width), int(frame_height)))

    return cap, frame_height, frame_width, writer


def video_init_thread(is_2_write = False, save_path = None):


    writer = None

    
    vs = WebcamVideoStream(src = 1).start()  # video input selection: 0 for built-in camera, 1 for Logitech C170


       
    if is_2_write is True:
        
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(frame_width), int(frame_height)))

    return vs, writer

def detect_MTCNN_webcam():

    # solve_cudnn_error()

    detector = MTCNN()
    #print('MTCNN detector loaded\r\n\r\n\r\n')

    
    print('Select input source: (1) webcam (2) photo in directory')
    input_source = input()

    if input_source == '1': # image input from webcam

        with tf.Graph().as_default():
            config = tf.compat.v1.ConfigProto(log_device_placement = True,
                                              allow_soft_placement = True,  # 允許當找不到設備時自動轉換成有支援的設備
                                              )
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.7
            
            sess = tf.compat.v1.Session(config = config)

        
        #----var
        frame_count = 0
        FPS = "Initialising"
        no_face_str = "No faces detected"

        # bounding boxes and prob
        box_colour = (0, 255, 0)
        font = cv2.FONT_HERSHEY_COMPLEX


        #----video streaming init
        # cap, frame_height, frame_width, writer = video_init(is_2_write = False) # for cv2.VideoCapture()
        vs, writer = video_init_thread(is_2_write = False) # for WebcamVideoStream()
        

        while (True): # original: while (cap.isOpened()) for cv2.VideoCapture()
            
            #----get image: Capture frame-by-frame
            #ret, img = cap.read() # ret is True if 
            
            #----get image: WebcamVideoStream() method
            ret = True
            img = vs.read()
            
            #----adjust frame size
            img = imutils.resize(img, width = 300)

            #----image processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if ret is True: # an image is successfully captured

                # detect faces by MTCNN
                results = detector.detect_faces(img_rgb)
                # print('img detected')


                # extract faces

                extract_info = []

                n_faces = len(results) # how many faces got detected in the current frame
                #print(str(n_faces) + " faces detected.")

                if n_faces > 0: # MTCNN detected faces

                    for i in range(n_faces):

                        x1, y1, width, height = results[i]['box']
                        x1, y1 = abs(x1), abs(y1)
                        x2, y2 = x1 + width, y1 + height

                        face = img_rgb[y1:y2, x1:x2]

                        #prob = results[i]['confidence']
                        #prob = truncate(prob, 3) # truncate the confidence to 3 decimals
                        #prob_str = '(' + str(prob) + ')'

                        # draw a rectangle on each face
                        
                        img_rgb = cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # write the confidence above the bounding box

                        #org = ( int( 3 * x1 / 4 + x2 / 4), int(y1 - 20) ) # coordinates of the bottom-left corner of the text string
                        #img_rgb = cv2.putText(img_rgb, prob_str, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                        # rsize pixels to the model size
                        image = Image.fromarray(face)
                        image = np.resize(image, required_size)
                        face_array = asarray(image)

                        # store to detect_info
                        
                        face_dict = {
                            'face_array': face_array,
                            'bounding_box': [x1, y1, width, height]
                            #'prob': prob
                            }
                        extract_info.append(face_dict)
                        
                   
                else: # no face detected
                    cv2.putText(img_rgb, no_face_str, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                
                #----FPS count
                if frame_count == 0:
                    t_start = time.time()
                frame_count += 1

                if frame_count >= 10:
                    FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                    frame_count = 0

                # show FPS info on the frame
                cv2.putText(img_rgb, FPS, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                #----image display
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                cv2.imshow("detect faces by MTCNN- Friedrich", img_rgb)

                
                #----image writing
                if writer is not None:
                    writer.write(img_rgb)

                #----'q' key pressed?
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                             
            else:
                print("failed to get images")
                break
           

        #----release the camera
        #cap.release() # for cv2.VideoCapture()
        vs.stop() # for WebcamVideoStream()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    elif input_source == '2': # image input from dir

        print('image of multiple faces')
        img = cv2.imread(r"E:\AI_Engineer_Portfolio_Projects\Face_Detection_MTCNN_Keras\yuming.jpg") 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect faces by MTCNN
        results = detector.detect_faces(img_rgb)
        print('img detected')


        # extract the faces

        extract_info = []

        n_faces = len(results) # how many faces got detected

        for i in range(n_faces):
            x1, y1, width, height = results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            face = img_rgb[y1:y2, x1:x2]

            prob = results[i]['confidence']
            prob = truncate(prob, 3) # truncate the confidence to 3 decimals
            prob_str = '(' + str(prob) + ')'

            # draw a rectangle on each face
            colour = (0, 255, 0)
            img_rgb = cv2.rectangle(img_rgb,
                                    (x1, y1),
                                    (x2, y2),
                                    color = colour,
                                    thickness = 3
                                    )

            # write the confidence above the bounding box

            org = ( int( 3 * x1 / 4 + x2 / 4), int(y1 - 20) ) # coordinates of the bottom-left corner of the text string
            fontScale = height / 250.
            img_rgb = cv2.putText(img_rgb, prob_str, org, cv2.FONT_HERSHEY_COMPLEX, fontScale, colour, 2)


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
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow(file_name, img_rgb)
        key = cv2.waitKey(0)

        if key == ord('s'): # 's' for 'save'
            cv2.imwrite(file_name, img_rgb)

        print('\r\n\r\n\r\n')

    else: # failed to input any image
        print('Invalid input source!')


required_size = (160, 160) # according to paper



if __name__ == '__main__':

    detect_MTCNN_webcam()
