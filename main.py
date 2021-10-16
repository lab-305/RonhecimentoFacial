import cv2 as opcv
import frame_convert2
import freenect

import numpy as np

import os
from random import shuffle
from tqdm import tqdm
 
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import matplotlib.pylot as plt

# genering data set
def generate_dataSet():
    face_classifier = opcv.CascadeClassifier("haar_face.xml")
    def face_cropped(img):
        gray = opcv.cvtColor(img, opcv.COLOR_BAYER_BG2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    cap = opcv.VideoCapture(0)                                          # camara PC
    # cap = frame_convert2.video_cv(freenect.sync_get_video()[1])       # kinect
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = opcv.resize(face_cropped(frame), (200,200))
            face = opcv.cvtColor(face, opcv.COLOR_BGR2GRAY)
            file_name_path = "fotos/"+"mike."+str(img_id)+".jpg"
            opcv.imwrite(file_name_path, face)
            opcv.putText(face, str(img_id), (50,50), opcv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            opcv.imshow("Cropped_face", face)
            if opcv.waitKey(1)==13 or int(img_id)==100:
                break

    cap.realise()
    opcv.destroyAllWindows()
    print("coleção de dados completo!!!")

generate_dataSet()

# create label
def my_label(image_name):
    name = image_name.split('.')[-3]
    if name == "mike":
        return np.array([1,0])

# create data 
def my_data():
    data =[]
    for img in tqdm(os.listdir("fotos")):
        path = os.path.join("data", img)
        img_data = opcv.imread(path, opcv.IMREAD_GRAYSCALE)
        img_data = opcv.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    
    shuffle(data)
    return data

data = my_data();

train = data[:24]
test = data[24:]

X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50,1)
print(X_train.shape)

Y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
print(X_test.shape)
Y_test = [i[1] for i in test]

# creating the model
tf.reset_default_graph()
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu ')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 12, 5, activation='relu ')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu ')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu ')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activition='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activition='softmax')
convnet = regression(convnet, optmizer='adam', learning_rate=0.001, loss='categorical_cossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.fit(X_train, Y_train, n_epoch=12, validation_set=(X_test, Y_test), show_metric= True,run_id="FRS")

# # visualize the data and make prediction
# def data_for_visualaztion():
#     vdata = []
#     for img in tqdm(os.listdir("images for visulization")):
#         path = os.path.join("images for visulization", img)
#         img_num = img.split('.')[0]
#         img_data = opcv.imread(path, opcv.IMREAD_GRAYSCALE)
#         img_data = opcv.resize(img_data, (50,50))
#         vdata.append([np.array(img_data), img_num])
#     shuffle(vdata)
#     return vdata

# vdata = data_for_visualaztion()

# fig = plt.figure(figsize=(20,20))
# for num, data in enumerate(vdata[:20]):
#     img_data = data[0]
#     y = fig.add_subplot(5,5, num+1)
#     image = img_data
#     data = img_data.reshape(50,50,1)
#     model_out = model.predict([data])[0]

#     if np.argmax(model_out) == 0:
#         my_label = 'mike'
    
#     y.imshow(image, cmap='gray')
#     plt.title(my_label)

#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_xaxis().set_visible(False)

# plt.show()
