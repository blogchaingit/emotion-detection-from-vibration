import RPi.GPIO as GPIO
import os
import smbus
import time
import pygame.mixer
import time
import requests
import cv2 
import os
import numpy as np

#This section is for the accelerometer to detect the vibration
class ADXL345():
	DevAdr = 0x53
	myBus = ""
	if GPIO.RPI_INFO['P1_REVISION'] == 1:
		myBus = 0
	else:
		myBus = 1
	b = smbus.SMBus(myBus)

	def setUp(self):
		self.b.write_byte_data(self.DevAdr, 0x2C, 0x0B) # BandwidthRate
		self.b.write_byte_data(self.DevAdr, 0x31, 0x00) # DATA_FORMAT 10bit 2g
		self.b.write_byte_data(self.DevAdr, 0x38, 0x00) # FIFO_CTL OFF
		self.b.write_byte_data(self.DevAdr, 0x2D, 0x08) # POWER_CTL Enable

	def changeme(self):
		return self.getValue(0x32)

	def getValueY(self):
		return self.getValue(0x34)

	def getValueZ(self):
		return self.getValue(0x36)

	def getValue(self, adr):
		tmp = self.b.read_byte_data(self.DevAdr, adr+1)
		sign = tmp & 0x80
		tmp = tmp & 0x7F
		tmp = tmp<<8
		tmp = tmp | self.b.read_byte_data(self.DevAdr, adr)
#		print '%4x' % tmp # debug

		if sign > 0:
			tmp = tmp - 32768

		return tmp

#	tmp = self.b.read_word_data(self.DevAdr, adr)

myADXL345 = ADXL345()
myADXL345.setUp()
#ox = 0
#oy = 0
z = myADXL345.getValueZ()
oz = z
# LOOP
for a in range(1000):
    #x = myADXL345.changeme()
   # y = myADXL345.getValueY()
    z = myADXL345.getValueZ()
    os.system("clear")
   # print("X=", x)
    #print("Y=", y)
    print("Z=", z)
    #if abs(x-ox) >= 10:
     #   print("Earthquake!")
    #if abs(y-oy) >= 10:
       # print("Earthquake!")
    if abs(z-oz) >= 30:
        print("Earthquake!")
    #ox = x
    #oy = y
        break
    oz = z
	
    time.sleep(1)
    
#################################################################################
    
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
i = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if i<=2 :
        img_name = "test{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        i = i+1
    else :
        break


cam.release()

cv2.destroyAllWindows()
    
    
    
subjects = ['','Fear','Not fear']
subject = ['', 'Test1', 'Test2']

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    global label_text
    img = test_img.copy()
    face, rect = detect_face(img)
    print(face)
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]
    print(label_text)
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

print("Predicting images...")

test_img1 = cv2.imread("test-data/test3.jpg")
#test_img2 = cv2.imread("test-data/test4.jpg")

predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)
print("Prediction complete")

cv2.imshow(subject[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subject[2], cv2.resize(predicted_img2, (400, 500)))

#############################################################################

pygame.mixer.init()


#print("input!")
#inputNum=input()

token ='zupAswTpvVzQO0gPv2EIC6xPtgZfjAiIATtQnsgqJO7'

url = 'https://notify-api.line.me/api/notify'
headers = {'Authorization':'Bearer '+token}
if label_text == 'Fear':
    payload = {'message':'The vibration occurs, user is fear, DANGER!'}
    res = requests.post(url,data=payload,headers=headers)
    print('res')
    pygame.mixer.music.load("jishin.mp3")
    pygame.mixer.music.play(-1)
    time.sleep(10)
    pygame.mixer.music.stop()

elif label_text== 'Not fear':
    payload = {'message':'The vibration occurs,but user is not fear, It is OK !'}
    res = requests.post(url,data=payload,headers=headers)
    pygame.mixer.music.load("song2.mp3") 
    pygame.mixer.music.play(-1)
    time.sleep(10)
    pygame.mixer.music.stop()
    
else:
    print("no music")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
   # payload = {'message':'no dangerous'}
   # res = requests.post(url,data=payload,headers=headers)
