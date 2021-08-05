import cv2

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")

width, height = 200, 200
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    imgcinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = detectorFace.detectMultiScale(imgcinza,
                                                     scaleFactor=1.5,
                                                     minSize=(100, 100))

    for (x,y,w,h) in faces_detectadas:
        img_redm = cv2.resize(imgcinza[y:y+h, x:x+w], (width, height))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        id, accuracy = reconhecedor.predict(img_redm)
        if id == 1:
            nome = "Laura"
        elif id  == 2:
            nome = "Eduardo"
        else:
            nome = "Eustaquio"

        cv2.putText(frame, nome, (x, y + (h+30)), font, 3, (255,0,0))


    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()