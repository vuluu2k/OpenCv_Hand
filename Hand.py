import cv2
import time
import os
import HandItem as htm

cap = cv2.VideoCapture(0)
pTime = 0

FolderPath = "Fingers"
lst = os.listdir(FolderPath)
lst_path = [cv2.imread(f"{FolderPath}/{item}") for item in lst]

detector = htm.handDetector(detectionCon=1)

while True:
    ret, frame = cap.read()
    frame_flip = cv2.flip(frame, 1)
    frame_flip = detector.findHands(frame_flip)
    nodeHands = detector.findPosition(frame, draw=False)

    fingers = []
    if len(nodeHands) != 0:
        # height hand length
        for i in range(6, 21, 4):
            if nodeHands[i + 2][2] < nodeHands[i][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        if nodeHands[4][1]>nodeHands[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    fingerCount = fingers.count(1)

    height, width, channel = lst_path[fingerCount].shape

    frame_flip[0:height, 0:width] = lst_path[fingerCount]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame_flip, f"FPS:{int(fps)}", (width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("handle Hand in Python with Opencv", frame_flip)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
