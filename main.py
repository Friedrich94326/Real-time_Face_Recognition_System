from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from numpy import asarray
import time
import tensorflow as tf
import imutils # to fasten FPS
from truncate import *
from keras.models import load_model

from detect_faces import *
from train_SVM_classifier import * # load data and train SVM model


def video_init(is_2_write = False, save_path = None):
   
    cap = cv2.VideoCapture(0) # video input selection: 0 for built-in camera, 1 for my Logitech C170
    
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    
    if is_2_write is True:
        
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'results/demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 10, (int(frame_width), int(frame_height))) # reference fps = 24
        
    return cap, writer

def detect_MTCNN_webcam():
    
    required_size = (160, 160, 3)
    detector = MTCNN()
    
    # load a pre-trainded FaceNet network- May refer to David Sandberg's model
    FaceNet_model = load_model('keras-facenet/model/facenet_keras.h5')


    with tf.Graph().as_default():
        config = tf.compat.v1.ConfigProto(
            log_device_placement = True,
            allow_soft_placement = True
            )
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
            
        sess = tf.compat.v1.Session(config = config)

        
    #----var
    frame_count = 0
    FPS = "Initialising"
    no_face_str = "No faces detected"

    

    #----video streaming init
    cap, writer = video_init(is_2_write = False) # for cv2.VideoCapture()
   
        

    while cap.isOpened():
            
        #----get image: Capture frame-by-frame
        ret, img = cap.read()

        #----image processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if ret is True: # an image is successfully captured

            results = detector.detect_faces(img_rgb)
            
            
            # extract faces
            n_faces = len(results) # how many faces got detected in the current frame

            if n_faces > 0:

                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # draw a rectangle on each face  
                img_rgb = cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # rsize pixels to the model size
                face = img_rgb[y1:y2, x1:x2, :]
                image = Image.fromarray(face)
                image = np.resize(image, required_size)
                face_array = asarray(image)
                    
                    
                # Calculate face embedding
                face_emb = get_embedding(FaceNet_model, face_array)
    
    
                # Predict the face- SVM_clf trained before
                samples = expand_dims(face_emb, axis = 0)
                yhat_class = SVM_clf.predict(samples)
                yhat_prob = SVM_clf.predict_proba(samples)

                # Get the name
                class_index = yhat_class[0]
                class_probability = yhat_prob[0, class_index] * 100
                predicted_names = out_encoder.inverse_transform(yhat_class)
    
                # Display the predicted label and probability
                title = '%s (%.3f %%)' % (predicted_names[0], class_probability)
                cv2.putText(img_rgb, title, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                        
                   
            else: # no face detected
                cv2.putText(img_rgb, no_face_str, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            
                
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
    cap.release()
    if writer is not None:
        writer.release()
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_MTCNN_webcam()
