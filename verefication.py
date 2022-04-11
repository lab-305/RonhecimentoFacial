# import frame_convert2
# import freenec
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

#       3 - Load and Preprocess images

# preprocessing - scale and resize
def preprocess (img_path):
    # print('preprocessing...')
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img/255.0
    return img

#       4 - Model engineering

# build distance layer
# Siamese L1 Distance class
class L1Dist(Layer):
    # print('L1Dist...')
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

#       7 - Save  model

# Reload model
# print('Reload model...')
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

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

# 0.9, 0.7
results, verified = verify(model, 0.3, 0.5)
print(verified)

# real time verification with OpenCV
# def real_time_verificati on():
#     # print('real time verification...')
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame = frame[120:120+250, 200:200+250, :]

#         cv2.imshow('Verification', frame)

#         # verification trigger
#         if cv2.waitKey(10) & 0xFF == ord('v'):

#             # save input image to application_data/input_input_folder
#             cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

#             # run verification
#             results, verified = verify(model, 0.9, 0.7)
#             print(verified)
        
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# real_time_verification()
