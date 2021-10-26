import cv2 as opcv
import frame_convert2
import freenect

# genering data set
def generate_dataSet():
    face_classifier = opcv.CascadeClassifier("haar_face.xml")
    
    def face_cropped(img):
        gray = opcv.cvtColor(img, opcv.COLOR_BGR2BGRA)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    img_id = 0
    while True:
        frame = frame_convert2.video_cv(freenect.sync_get_video()[0])  # kinect
        if face_cropped(frame) is not None:
            img_id += 1

            face = opcv.resize(face_cropped(frame), (200, 200))
            face = opcv.cvtColor(face, opcv.COLOR_BGR2GRAY)

            file_name_path = "data/" + "name." + str(img_id) + ".jpg"
            opcv.imwrite(file_name_path, face)
            opcv.putText(face, str(img_id), (50, 50), opcv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            opcv.imshow("Cropped_face", face)
            if opcv.waitKey(1)==13 or img_id==100:
                break

    print("coleção de dados completo!!!")

generate_dataSet()
