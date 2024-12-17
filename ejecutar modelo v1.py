
import cv2
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)
cap.set(3, 960)  
cap.set(4, 720) 

model = YOLO(r'C:\Users\ERICF\Documents\Base_de_datos\Python\Model\models\modelo4.pt')

while True:
    ret, frame = cap.read()

    results = model.predict(frame, imgsz=640, conf=0.6)
    

    person_count = len(results[0].boxes) if results and results[0].boxes else 0

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] * 100  

            if confidence > 50:  
                color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  
                cv2.putText(frame, f'{confidence:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2) 
                
    cv2.putText(frame, f'Personas detectadas: {person_count}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  

    cv2.imshow('Persona detectada', frame)

    t = cv2.waitKey(5)
    if t == 27:
        break
    
    time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
