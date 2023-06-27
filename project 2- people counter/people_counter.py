from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('../Videos/people.mp4')
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
mask = cv2.imread('mask.png')

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits_up = [103, 161, 296, 161]
limits_down = [527, 489, 735, 489]

total_count_up = []
total_count_down = []

# ptime = 0
while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)

    img_graphics = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, img_graphics, (730, 260))
    results = model(img_region, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)

            w, h = x2 - x1, y2 - y1
            # bbox = int(x1),int(y1),int(w),int(h)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]

            if current_class == 'person' and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=8)
                # cvzone.putTextRect(img, f"{current_class} {conf}", (max(0, x1), max(30, y1)),
                #                    scale=1, thickness=1, offset=3)
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    # ctime = time.time()
    # fps = int(1/(ctime-ptime))
    # ptime = ctime
    # cv2.putText(img,f'{fps}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    results_tracker = tracker.update(detections)

    cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 5)
    cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(30, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        if limits_up[0] < cx < limits_up[2] and limits_up[1] - 15 < cy < limits_up[1] + 15:
            if total_count_up.count(id) == 0:
                total_count_up.append(id)
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)

        if limits_down[0] < cx < limits_down[2] and limits_down[1] - 15 < cy < limits_down[1] + 15:
            if total_count_down.count(id) == 0:
                total_count_down.append(id)
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 5)
    #
    # # cvzone.putTextRect(img, f" count: {len(total_count)}", (50, 50))
    cv2.putText(img, str(len(total_count_up)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 7)
    cv2.putText(img, str(len(total_count_down)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 7)

    cv2.imshow('video', img)
    # cv2.imshow('Img region', img_region)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
