from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture('../Videos/cars.mp4')
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

ptime = 0
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
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            cls = box.cls[0]
            cvzone.putTextRect(img, f"{classNames[int(cls)]} {conf}", (max(0, x1), max(30, y1)),scale=1,thickness=1)



    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(img,f'{fps}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow('video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

