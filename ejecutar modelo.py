import cv2
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1080)  
cap.set(4, 720)   

model = YOLO(r'C:\Users\ERICF\Documents\Base_de_datos\Python\Model\models\modelo4.pt')

while True:
    ret, frame = cap.read()

    results = model.predict(frame, imgsz=640, conf=0.6)

    person_count = len(results[0].boxes) if results and results[0].boxes else 0

    annotated_frames = results[0].plot()

    cv2.putText(annotated_frames, f'Personas detectadas: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow('Persona detectada', annotated_frames)

    t = cv2.waitKey(5)
    if t == 27:
        break
    
    time.sleep(0.05)
    
cap.release()
cv2.destroyAllWindows()  
