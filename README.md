# **Portfolio Project: Real-time Face Recognition System Using Keras**


## **Flow Chart of the Face Recognition System**

![Image of Flow Chart](https://github.com/Friedrich94326/Portfolio_Project/blob/main/Overall%20Flow%20Chart.png)

## Stage Management

***Phase*** | ***Description***
---- | ---
Build datasets | 5_celebrities_dataset <br> family_and_friends <br/>
Build a face detector | MTCNN-based detector to draw bounding box for each face detected and even crop the face
Face Embedding | Use FaceNet's inception network to get face embeddings for representing each face captured <br/> Transfer learning: fine-tuning the pre-trained FaceNet model <br/>
Build a face classifier | Using SVM to classify face embeddings as one of faces in our dataset
Integrate system | Integrate detector and classifier into the entire recognition system
Display control | Use OpenCV VideoCapture() to receive video stream <br> Display bounding box, predicted label, and its probability of each face on the screen <br/>
Compare other models | Use other common pre-trained neural networks (e.g., VGG-16, DeepFace, Haar cascade) to perform our task



## DEMO

<p float="left">
< img src="https://github.com/Friedrich94326/Portfolio_Project/blob/main/results/Predictions_Friedrich.png" width="240" />
< img src="https://github.com/Friedrich94326/Portfolio_Project/blob/main/results/Predictions_Friedrich_2.png" width="240"/ >
  </p>


## **Development Tools:**
TensorFlow-GPU: version 2.3.0 \
Keras: version 2.4.3 \
OpenCV: version 4.2.0 \
Python: version 3.6.9 \
FaceNet's Inception Model \
MTCNN \
Scikit-learn: version 0.23.2 \
Kalman Filter \
GUI \
Logitech C170 Webcam \
CUDA version 11.1



## **References:**
(1) [《FaceNet: A Unified Embedding for Face Recognition and Clustering》](https://arxiv.org/abs/1503.03832)  \
(2) [《DeepFace: Closing the gap to human-level performance in face verification》](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf) \
(3) &nbsp; David Sandberg's prominent project: *[Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet)* \
(4) &nbsp; *[MTCNN](https://github.com/ipazc/mtcnn)* for face detection \
(5) [《Face Detection in Python Using a Webcam》](https://realpython.com/face-detection-in-python-using-a-webcam/) \
(6) &nbsp;[Transfer Learning and Fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning?fbclid=IwAR1h325gQ5L5S-QCwrlZWhsPE5qm4XGUsrsLJdjppzs_RABCOhxARn8voKA) \
(7) [《Face Recognition: Real-Time Face Recognition System using Deep Learning Algorithm and Raspberry Pi 3B》](https://medium.com/@BhashkarKunal/face-recognition-real-time-webcam-face-recognition-system-using-deep-learning-algorithm-and-98cf8254def7) \
(8) &nbsp; Chapter 14- Face Recognition [*Digital Image Processing: An Algorithmic Approach with MATLAB*](https://www.amazon.com/Digital-Image-Processing-Algorithmic-Textbooks/dp/1420079506), Uvais Qidwai and <br> &nbsp; &nbsp; &nbsp; &nbsp; C.H. Chen \
(9) &nbsp; Dr. Jason Brownlee's article on [developing a face recognition system using FaceNet model in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) 
