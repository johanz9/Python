
"""
Dataset create N images of the person

"""

import cv2

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

count = 0

video = cv2.VideoCapture("shun.mp4")


print("Starting Creating Dataset")
while(True):
    _, imagen = video.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 4)
        count += 1
        print(count)

        cv2.imwrite("images/Shun_Oguri/Y_" + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow("Creating Dataset", imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif count >= 400:
        video.release()
        break


print("Creating Dataset Finished")
video.release()
cv2.destroyAllWindows()