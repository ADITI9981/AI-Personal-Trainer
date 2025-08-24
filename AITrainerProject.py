import cv2
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture("AITrainer/curls.mp4")

detector = pm.poseDetector()
count=0
dir = 0
pTime =0

while True:
    success, img = cap.read()
    img = cv2.resize(img,(1200,720))
    #img = cv2.imread("AITrainer/test.png")
    img = detector.findPose(img,False)
    lmList = detector.findPosition(img,False)
    #print(lmList)

    if len(lmList) !=0:
        #rightarm
        #detector.findAngle(img,12,14,16)
        #left arm
        angle = detector.findAngle(img,11,13,15)
        per = int(np.interp(angle, (210, 310), (0, 100)))
        bar = int(np.interp(angle, (210, 310), (400, 100)))
        #print(angle,per)

        #check for the dumbell curls
        color = (255,0,255)
        if per == 100:
            color = (0,255,0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per ==0:
            color = (0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(img.shape)

        # progress bar outline
        cv2.rectangle(img, (540, 100), (600, 400), (255, 0, 255), 3)  # outline
        cv2.rectangle(img, (540, bar), (600, 400), (255, 0, 255), cv2.FILLED)  # fill
        cv2.putText(img, f'{per} %', (450, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # --- Curl counter (left side) ---
        cv2.rectangle(img, (20, 250), (180, 450), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (50, 380), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 12)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(50,100), cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),5)
    cv2.imshow("Image",img)
    cv2.waitKey(1)