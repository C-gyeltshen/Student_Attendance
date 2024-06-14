import cv2
import pickle 
import numpy as np
import os

def capture_faces(name, max_samples=100, save_path='data'):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    faces_data = []
    i = 0

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < max_samples and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == max_samples:
            break
    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(-1, 50 * 50 * 3)
    
    names_path = os.path.join(save_path, 'names.pkl')
    faces_path = os.path.join(save_path, 'faces_data.pkl')

    if 'names.pkl' not in os.listdir(save_path):    
        names = [name] * len(faces_data)
        with open(names_path, 'wb') as f:       
            pickle.dump(names, f)
    else:    
        with open(names_path, 'rb') as f:
            names = pickle.load(f)    
            names += [name] * len(faces_data)
        with open(names_path, 'wb') as f:   
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir(save_path):
        with open(faces_path, 'wb') as f:       
            pickle.dump(faces_data, f)
    else:   
        with open(faces_path, 'rb') as f:
            faces = pickle.load(f)
            # Reshape existing faces data to match the shape of faces_data along dimension 1
            faces = faces.reshape(-1, 50 * 50 * 3)
            faces = np.append(faces, faces_data, axis=0)
        with open(faces_path, 'wb') as f:        
            pickle.dump(faces, f)

if __name__ == "__main__":
    name = input("Enter Your Name: ")
    capture_faces(name)
