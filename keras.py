
import tensorflow.keras as keras
import numpy 
import time
import cv2
import mss
import os
import sys
from database import db
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
title = "uji coba screen graber"
start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 80, "left":500, "width": 448, "height":448}

numpy.set_printoptions(suppress=True)
#webcam = cv2.VideoCapture(0)
model = keras.models.load_model('keras_model.h5')
data_for_model = numpy.ndarray(shape=(1, 224, 224, 3), dtype=numpy.float32)

def load_labels(path):
	f = open(path, 'r')
	lines = f.readlines()
	labels = []
	for line in lines:
		labels.append(line.split(' ')[1].strip('\n'))
	return labels

label_path = 'labels.txt'
labels = load_labels(label_path)
print(labels)

# This function proportionally resizes the image from your webcam to 224 pixels high
def image_resize(image, height, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# this function crops to the center of the resize image
def cropTo(img):
    size = 224
    height, width = img.shape[:2]
    sideCrop = (width - 224) // 2
    return img[:,sideCrop:(width - sideCrop)]



def screen_recordMSS():
    global fps, start_time
    while True:
        #success,img = sct.grab(monitor)
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = image_resize(img, height=224)
        img = cropTo(img)
        # To get real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = numpy.expand_dims(img, axis=0)
        # Display the picture
        img2 = numpy.array(sct.grab(monitor))
                                 
    #print(webcam.read())
        #if ret:
        normalized_img = (img.astype(numpy.float32) / 127.0) - 1
        data_for_model[0] = normalized_img
        
            #run inference
        prediction = model.predict(data_for_model)
        for i in range(0, len(prediction[0])):
            #a = float(prediction[0][0])
            #b = float(prediction[0][1])
            #print('{}: {}'.format(labels[i], prediction[0][1]))
            a = float(prediction[0][0])
            b = float(prediction[0][1])
            c = float(prediction[0][2])
            d = float(prediction[0][3])
            #print (a)
            print(prediction)
           #time.sleep(1)
            if a  > 0.9 :
                cv2.putText(img2, ' label: ' + str(labels[0]),(0, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                cv2.putText(img2, ' predict: ' + str(prediction[0][0]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2) 
                print(' predict: ' + str(labels[0]))
                #db.child("suara").set("1")

            elif b >0.9:
                cv2.putText(img2, ' label: ' + str(labels[1]),(0, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                cv2.putText(img2, ' predict: ' + str(prediction[0][1]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2) 
                print(' predict: ' + str(labels[1]))
                db.child("suara").set("3")
            elif c >0.9:
                cv2.putText(img2, ' label: ' + str(labels[2]),(0, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                cv2.putText(img2, ' predict: ' + str(prediction[0][2]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2) 
                print(' predict: ' + str(labels[2]))
                db.child("suara").set("4")
            
            elif d >0.9:
                cv2.putText(img2, ' label: ' + str(labels[3]),(0, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                cv2.putText(img2, ' predict: ' + str(prediction[0][3]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2) 
                print(' predict: ' + str(labels[3]))
                db.child("suara").set("0")
            '''
            else :
                cv2.putText(img2, ' label: ' +'idle',(0, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
                #cv2.putText(img2, ' predict: ' + str(prediction[0][2]),(0, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2) 
                #print(' predict: ' + str(labels[2]))
            '''
        fps+=1

        TIME = time.time() - start_time
        if (TIME) >= display_time :
            #print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        cv2.imshow(title, img2)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

screen_recordMSS()
