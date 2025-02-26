# Importamos librerias
import cv2
from ultralytics import YOLO
import time

# Ejecutamos el modelo y dimensionamos el tamaño de la pestaña que se mostrara
cap = cv2.VideoCapture(0)
cap.set(3, 960)  
cap.set(4, 720) 

# Ponemos la ruta del modelo .pt en este caso usamos el modelo4.pt
model = YOLO(r'ErickFler/Model/models/modelo4.pt')

# Creamos un bucle
while True:
    ret, frame = cap.read()
# Generamos el grado de confianza
    results = model.predict(frame, imgsz=640, conf=0.6)
    
# Realizamos el contador de las personas en cuadro
    person_count = len(results[0].boxes) if results and results[0].boxes else 0

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] * 100  

            # En este punto ajustamos los colores en base a el grado de confianza
            if confidence > 50:  
                color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
                cv2.putText(frame, f'{confidence:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2) 
    # Generamos el conteo y lo imprimimos en la pestaña           
    cv2.putText(frame, f'Personas detectadas: {person_count}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  

    cv2.imshow('Persona detectada', frame)

    #Usamos la tecla espaciadora para salir de la camara, de lo contrario no se detendra
    t = cv2.waitKey(5)
    if t == 27:
        break
    
    # Ajustamos el tiempo entre cada ciclo
    time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
