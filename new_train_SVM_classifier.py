from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# Load test images
test_img_data = load('Intermediate_Data/family_and_friends_LITE_images.npz')
Test_Imgs = test_img_data['arr_0']

# Load face arrays
data = load('Intermediate_Data/family_and_friends_LITE_faces_dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# Load face embeddings
data = load('Intermediate_Data/family_and_friends_LITE_face_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# Normalise input vectors
in_encoder = Normalizer(norm = 'l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)


# Fit SVM classifier
SVM_clf = SVC(kernel = 'linear', probability = True)
SVM_clf.fit(trainX, trainy)

# Predict
yhat_train = SVM_clf.predict(trainX)
yhat_test = SVM_clf.predict(testX)

# Score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

# Summarise
print('Accuracy: train = %.3f %%, test = %.3f%%' % (score_train * 100, score_test * 100))