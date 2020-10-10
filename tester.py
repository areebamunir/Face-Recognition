import cv2
import os
import numpy as np
import FaceRecognition as fr


#Read image to perform face recognition
test_img=cv2.imread("C:\\Users\\HP\Desktop\\ML\\Face Detection\\Testing\\Chris7.jpeg")#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


#Comment these three lines while running this program second time. As it saves training.yml file in directory
faces,faceID=fr.labels_for_training_data('C:\\Users\\HP\\Desktop\\ML\\Face Detection\\Training')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')


#TRAINING CLASSIFIER
# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#SAVE CLASSIFIER
# face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

#Labelling for face
name={0:"Emma",1:"Chris"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):#Larger confidence than 37 then do not print predicted face text on the screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(600,600))#resize image axes
cv2.imshow("face dtecetion first trial",resized_img)
cv2.waitKey(0)#Waits until a key is pressed
cv2.destroyAllWindows


