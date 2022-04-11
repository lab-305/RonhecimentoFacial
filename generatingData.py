# import frame_convert2
# import freenec
import cv2
import os

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
