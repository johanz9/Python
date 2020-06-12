import cv2
import pickle

cascPath = "Cascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

eyeCascade = cv2.CascadeClassifier("Cascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("Cascades/haarcascade_smile.xml")

reconocimiento = cv2.face.LBPHFaceRecognizer_create()
reconocimiento.read("entrenamiento.yml")

etiquetas = {"nombre_persona" : 1 }
with open("labels.pickle",'rb') as f:
    pre_etiquetas = pickle.load(f)
    etiquetas = { v:k for k,v in pre_etiquetas.items()}

web_cam = cv2.VideoCapture("video3.mp4")

width = web_cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = web_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
print('width, height:', width, height,False)

fps = web_cam.get(cv2.CAP_PROP_FPS)
print('fps:', fps)  # float
# print(cv2.CAP_PROP_FPS) # 5

frame_count = web_cam.get(cv2.CAP_PROP_FRAME_COUNT)
print('frames count:', frame_count)  # float
# print(cv2.CAP_PROP_FRAME_COUNT) # 7


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, fps, (int(width),int(height)))

while True:
    # Capture el marco
    ret, marco = web_cam.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

    if ret:
        # The frame is ready and already captured
        pos_frame = web_cam.get(cv2.CAP_PROP_POS_FRAMES)
        print (str(pos_frame) + " frames")
    else:
        # The next frame is not ready, so we try to read it again
        web_cam.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
        print ("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)
        out.release()
        break


    grises = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
    rostros = faceCascade.detectMultiScale(grises, 1.5, 5)



    # Dibujar un rectángulo alrededor de las rostros
    for (x, y, w, h) in rostros:
        #print(x,y,w,h)
        roi_gray = grises[y:y+h, x:x+w]
        roi_color = marco[y:y+h, x:x+w]

        # reconocimiento
        id_, conf = reconocimiento.predict(roi_gray)
        if conf >= 4  and conf < 85:
            #print(id_)
            #print(etiquetas[id_])           
            font = cv2.FONT_HERSHEY_SIMPLEX            

            nombre = etiquetas[id_]

            if conf > 50:
                #print(conf)
                nombre = "Desconocido"

            color = (255,255,255)
            grosor = 2
            cv2.putText(marco, nombre, (x,y), font, 1, color, grosor, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 255, 0), 2)

        rasgos = smileCascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in rasgos:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Display resize del marco  
    marco_display = cv2.resize(marco, (1200, 650), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Detectando Rostros', marco_display)

    # write the flipped frame
    out.write(marco)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo está hecho, liberamos la captura
web_cam.release()
out.release()
cv2.destroyAllWindows()