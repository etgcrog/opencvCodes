import cv2

cam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("cascades//haarcascade_frontalface_default.xml")
amostra = 1
numeroAmostra = 25
id = input("Digite seu identificador: ")
width, height = 220, 220
print("Capturando as Faces")

while True:
    _, frame = cam.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(frame, scaleFactor=1.5,
                                        minSize=(150,150))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)
        if cv2.waitKey(1) & 0XFF == ord("q"):
            imagemface = cv2.resize(frame_gray[y:y+h, x:x+w], (width, height))
            cv2.imwrite(f"fotos/pessoa{str(id)}.{str(amostra)}.jpg", imagemface)
            print(f"Foto {str(amostra)} capturada com sucesso!")
            amostra += 1

    cv2.imshow("WebCam", frame)
    cv2.waitKey(1)
    if (amostra >= numeroAmostra + 1):
        break

print("Faces Capturadas com Sucesso")
cam.release()
cv2.destroyAllWindows()