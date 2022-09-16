import cv2
import HandItem as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Range volume
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]
print(volumeRange)

pTime = 0
cap = cv2.VideoCapture(0)

detector = htm.handDetector(detectionCon=1)

while True:
    ret, frame = cap.read()
    frame_flip = cv2.flip(frame, 1)
    frame_flip = detector.findHands(frame_flip)
    nodeHands = detector.findPosition(frame, draw=False)

    if len(nodeHands) > 0:
        x1, y1 = nodeHands[4][1], nodeHands[4][2]
        x2, y2 = nodeHands[8][1], nodeHands[8][2]
        x3, y3 = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw node for vol
        cv2.circle(frame_flip, (x1, y1), 4, (255, 255, 255), -1)
        cv2.circle(frame_flip, (x2, y2), 4, (255, 255, 255), -1)
        cv2.circle(frame_flip, (x3, y3), 4, (255, 255, 255), -1)
        cv2.line(frame_flip, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Use Pytago Math Line (x1,y1) (x2,y2)
        # line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_length = math.hypot(x2 - x1, y2 - y1)
        # length 15 -> 300

        # Covert length
        volumeConvertFromLine = np.interp(line_length, [15, 240], [minVolume, maxVolume])
        print(volumeConvertFromLine)
        volume.SetMasterVolumeLevel(volumeConvertFromLine, None)

        if line_length < 15:
            cv2.circle(frame_flip, (x3, y3), 4, (0, 0, 0), -1)

        volumeBar = np.interp(line_length, [15, 240], [200, 40])
        volumePercent = np.interp(line_length, [15, 240], [0, 100])
        cv2.rectangle(frame_flip, (10, 40), (50, 200), (255, 255, 255), 2)
        cv2.rectangle(frame_flip, (10, int(volumeBar)), (50, 200), (255, 255, 255), -1)
        cv2.putText(frame_flip, f'{round(volumePercent, 2)}%', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Show Camera', frame_flip)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
