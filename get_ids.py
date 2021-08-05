import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImgID():
    caminhos = [os.path.join("fotos", f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for img in caminhos:
        imggray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("faces", imagemface)
        # cv2.waitKey(1)
        id = int(os.path.split(img)[-1].split('.')[1])
        ids.append(id)
        faces.append(imggray)
    return np.array(ids), faces

ids, faces = getImgID()
# print(len(ids))
# print(faces)

print("TREINANDO!!!")

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')
"""
FOTOS FANTASMAS, EIGNEFACES, EXTRACAO DAS PRICIPAIS CARACTERISTICAS DAS PESSOAS
E COM BASE NELAS, FARA COMBINACAO LINEAR E SOMAR COM UMA MEAN IMAGE
eigenvector = vetor proprio 
eigenvalue = escalar multiplicado no vector
"""

# fisherface.train(faces, ids)
# fisherface.write("classificadorFisher.yml")
#
# lbph.train(faces,ids)
# lbph.write("classificadorLbph.yml")
print("Treinamento Realizado")



