import cv2

def collect_data():
    save_path = r'C:\Users\ERICF\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Tecnologico de Monterrey\BASES DE DATOS\Python\Model\data\datasets\images\example'
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)  
    cap.set(4, 720)   
    con = 0
    
    while True:
        ret, frame = cap.read()
        t = cv2.waitKey(5)  
        
        if t == 32:  
            cv2.imwrite(f'{save_path}\\img_example_{con}.jpg', frame)
            print(f"Imagen guardada: img_example_{con}.jpg")
            con += 1
        
        cv2.imshow('Person detect', frame)
        
        if t == 27:  
            break
    
    cap.release()
    cv2.destroyAllWindows()

collect_data()



