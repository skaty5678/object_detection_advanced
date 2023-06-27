from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture('../Videos/ppe-1.mp4')
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO('best.pt')

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat',
              'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest',
              'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan',
              'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader'
              ]

ptime = 0
my_color=(0,0,255)
while True:
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)

            w, h = x2-x1,y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            # cvzone.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            current_class = classNames[cls]
            if conf > 0.5:
                if current_class in ['Hardhat','Mask','Safety Vest']:
                    my_color = (0,255,0)
                elif current_class in ['NO-Hardhat','NO-Mask','NO-Safety Vest']:
                    my_color = (0,0,255)
                else:
                    my_color = (255,0,0)

                cvzone.putTextRect(img, f"{classNames[cls]} {conf}",
                                   (max(0, x1), max(30, y1)),scale=1,thickness=1,
                                   colorB=my_color,colorT=(255,255,255),colorR=my_color,offset=5)

                cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(img,f'{fps}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

