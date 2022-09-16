import cv2
import face_recognition

imgElon = face_recognition.load_image_file("pic/elon musk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgCheck = face_recognition.load_image_file("pic/elon check.jpg")
imgCheck = cv2.cvtColor(imgCheck, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 255, 255))

faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeElonCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck, (faceCheck[3], faceCheck[0]), (faceCheck[1], faceCheck[2]), (255, 255, 255))

results = face_recognition.compare_faces([encodeElon], encodeElonCheck)
faceDistance = face_recognition.face_distance([encodeElon], encodeElonCheck)

print(results, faceDistance)
cv2.putText(imgCheck, f"{results} {(1-round(faceDistance[0], 2))*100}%", (10, 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

cv2.imshow('Image', imgElon)
cv2.imshow('Image Check', imgCheck)
cv2.waitKey()
