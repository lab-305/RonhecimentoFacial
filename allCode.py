# import frame_convert2
# import freenec
import cv2

import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPool2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.ops.gen_array_ops import shape

# folders
# print('configuring folders path...')
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#       2 - Colecting and saving data/images
def registration():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        frame = frame[120:120+250, 200:200+250, :]

        if cv2.waitKey(1) & 0XFF == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
        
        if cv2.waitKey(1) & 0XFF == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
        
        cv2.imshow('Image Collection', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# registration()

#       3 - Load and Preprocess images

# get images directores
# print('geting imgs directories...')
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(30)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(30)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(30)

# preprocessing - scale and resize
def preprocess (img_path):
    # print('preprocessing...')
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img/255.0
    return img

# new we can choose any img from the anchor folder to see the result
# img = preprocess('data/anchor/029ee87c-4931-11ec-9672-e09467e2e08c.jpg')
# img.numpy().max()
# plt.imshow(img)

# Create labelled dataset
# labelled dataset will associate zeros to negatices img and ones to positive imgs
# print('create labelled datset...')
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()
example = samples.next()
# print('example: ', example)

# build train and test Partition
def preprocess_twin(input_img, validation_img, label):
    # print('preprocess_twin...')
    return (preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*example)
# plt.imshow(res[0])
# if we print res the lenght will be 3 (img1, img2, 1 if the imgs match or 0 if don´t )
# print('res: ', res)

# build data loader pipeline
# print('build data loader pipeline...')
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# samples = data.as_numpy_iterator()
# samp = samples.next()
# plt.imshow(samp[0])
# plt.imshow(samp[0])
# print('samp: ', samp[2])

# training partition
# print('training partition...')
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()
# print('len(train_sample[0]) ', len(train_sample[0]))

# testing partition
# print('testing partition...')
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

#       4 - Model engineering

# folowing the siamese model int the pdf file
def make_embedding():
    # print('make_embedding...')   
    inp = Input(shape=(100, 100, 3), name='input_name')

    # 1º bloco
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPool2D(64, (2,2), padding='same')(c1)

    # 2º bloco
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPool2D(64, (2,2), padding='same')(c2)

    # 3º bloco
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPool2D(64, (2,2), padding='same')(c3)

    # 4º bloco final embedding
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

# results
embedding = make_embedding()
# print('embedding ', embedding.summary())

# build distance layer
# Siamese L1 Distance class
class L1Dist(Layer):
    # print('L1Dist...')
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# Make siamese model
def make_siamese_model():
    # print('make_siamese_model...')

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))
    
    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # combine siamese distances componentes
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs = classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
# print('siamese_model ', siamese_model.summary())

#       5 - Training

# loss and optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# estabilish checkpoint
# print('estabilish checkpoint...')
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# build train step function
# test_batch = train_data.as_numpy_iterator()
# batch_1 = test_batch.next()

@tf.function
def train_step(batch):
    # print('train_step...')
    
    # record all of our operations
    with tf.GradientTape() as tape:
        # get anchor and positive/negative img
        x = batch[:2]
        # get label
        y = batch[2]
        # forward pass
        yhat = siamese_model(x, training=True)
        # calculate loss
        loss = binary_cross_loss(y, yhat)
    
    print('loss ', loss)

    # calculate gradientes
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # calculate update weihts and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

# training loop
def train(data, EPOCHS):
    # print('training loop...')

    # loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n EPOCH {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # loop through each batch
        for idx, batch in enumerate(data):
            # run train step here
            train_step(batch)
            progbar.update(idx+1)

        # save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# train the model
EPOCHS = 10
print('training the model...')
train(train_data, EPOCHS)

#       6 -  Evaluate  model

# get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# make predictions
y_hat = siamese_model.predict([test_input, test_val])
print('y_hat ', y_hat)

# Post processing the results
[1 if prediction > 0.5 else 0 for prediction in y_hat]
print('y_true ', y_true)

# calculate metrics
# creating a metric object
m = Recall()
# Calculating the recall value
m.update_state(y_true, y_hat)
# return recall result
m.result().numpy()
# creating a metric object
m = Precision()
# calculating the recall value
m.update_state(y_true, y_hat)
# Return recall result
m.result().numpy()

# Results
# set plot size
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1, 2 ,1)
# plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1, 2, 2)
# plt.imshow(test_val[0])

# Renders cleanly
# plt.show()
 
#       7 - Save  model

# Save weights
# print('Save weights...')
siamese_model.save('siamesemodel.h5')

# L1Dist

# Reload model
# print('Reload model...')
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
model.predict([test_input, test_val])
# print('model.summary ', model.summary())

#       8 - Verification  function

def verify(model, detection_thresold, verification_thresold):
    # print('Verification function...')

    # results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data','verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection threshold: Metric aboce wich a prediction is considered positive
    detection = np.sum(np.array(results) > detection_thresold)

    # verification threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_thresold

    return results, verified

# real time verification with OpenCV
def real_time_verification():
    # print('real time verification...')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250, 200:200+250, :]

        cv2.imshow('Verification', frame)

        # verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):

            # save input image to application_data/input_input_folder
            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

            # run verification
            results, verified = verify(model, 0.9, 0.7)
            print(verified)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time_verification()


# # genering data set
# cap = cv2.VideoCapture(0)
# def generate_dataSet():
#     face_classifier = cv2.CascadeClassifier("haar_face.xml")
    
#     def face_cropped(img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#         if faces is ():
#             return None
#         for (x, y, w, h) in faces:
#             cropped_face = img[y:y + h, x:x + w]
#         return cropped_face

#     img_id = 0
#     while True:
#         # frame = frame_convert2.video_cv(freenect.sync_get_video()[0])  # kinect
#         frame, n = cap.read();
        
#         if face_cropped(frame) is not None:
#             img_id += 1

#             face = cv2.resize(face_cropped(frame), (200, 200))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#             file_name_path = "data/" + "name." + str(img_id) + ".jpg"
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

#             cv2.imshow("Cropped_face", face)
#             if cv2.waitKey(1)==13: # or img_id==100:
#                 break

#     print("coleção de dados completo!!!")

# # cap.release()
# # cv2.destroyAllWindows()

# # generate_dataSet()
