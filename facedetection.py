# with function
import cv2
import csv
import os
import face_recognition
from PIL import Image
import numpy as np
import datetime
import geocoder
import time

def get_current_location():
    # Get the IP address of the system
    ip_address = geocoder.ip('me').ip

    # Use the IP address to get the location details
    location = geocoder.ip(ip_address)

    return location.address

def load_known_faces(csv_file_path):
    known_face_encoding = []
    known_faces_names = []

    with open(csv_file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            name = row[0]
            image_path = row[1]

            print(f"Name: {name}, Image Path: {image_path}")
            #image = Image.open(image_path)
            image = Image.open(image_path.strip('"'))
            face_encoding = face_recognition.face_encodings(np.array(image))[0]

            known_face_encoding.append(face_encoding)
            known_faces_names.append(name)

    return known_face_encoding, known_faces_names

def create_csv_file():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file_name = f"{current_date}.csv"

    with open(csv_file_name, mode='w', newline='') as csv_file:
        fieldnames = ['Name', 'Time', 'Date', 'Live Location']
        lnwriter = csv.writer(csv_file)
        lnwriter.writerow(fieldnames)

    return csv_file_name

def recognize_faces(known_face_encoding, known_faces_names, students, csv_file_name):
    face_capture = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    current_location = get_current_location()
    
    folder_mode_path = "C:/Users/hp/OneDrive/Desktop/FD/Resources/Modes"
    #C:\Users\hp\OneDrive\Desktop\FD\Resources\Modes
    mode_path_list = os.listdir(folder_mode_path)
    mode_image_list = []
    recognized_persons = {}  # Dictionary to store the time when a person was recognized
    

    for path in mode_path_list:
        mode_image_list.append(cv2.imread(os.path.join(folder_mode_path,path)))
    print("length = ",len(mode_image_list))

    
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,640)
    video_capture.set(4,480)

    imagebackground = cv2.imread("Resources/background.png")
    
    with open(csv_file_name, mode='a', newline='') as csv_file:
        lnwriter = csv.writer(csv_file)

        
        while True:
            
            ret, image_per_sec = video_capture.read()
            
            colour = cv2.cvtColor(image_per_sec, cv2.COLOR_BGR2GRAY)
            small_frame = cv2.resize(colour, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
            face_names = []

            imagebackground[162:162 + 480,55:55 +640] = image_per_sec
           
            imagebackground[44:44 + 633,808:808 +414] = mode_image_list[0]

            
            #time.sleep(3)
           

            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                face_distances = np.linalg.norm(known_face_encoding - face_encoding, axis=1)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                face_names.append(name)
                
                if name in known_faces_names:
                    
                    imagebackground[44:44 + 633,808:808 +414] = mode_image_list[2]


                    current_time = datetime.datetime.now().strftime("%H:%M:%S")

                    if name not in recognized_persons:
                        recognized_persons[name] = current_time
                        imagebackground[44:44 + 633, 808:808 + 414] = mode_image_list[2]
                    else:
                        # Check if 10 seconds have passed
                        recognition_time = datetime.datetime.strptime(recognized_persons[name], "%H:%M:%S")
                        elapsed_time = (datetime.datetime.strptime(current_time, "%H:%M:%S") - recognition_time).seconds
        
                        if elapsed_time >= 10:
                            imagebackground[44:44 + 633, 808:808 + 414] = mode_image_list[3]
                        else:
                            imagebackground[44:44 + 633, 808:808 + 414] = mode_image_list[2]

                    # if not already_appeared:
                    #     imagebackground[44:44 + 633,808:808 +414] = mode_image_list[2]
                    # else:
                        
                    #     imagebackground[44:44 + 633,808:808 +414] = mode_image_list[3]
                    if name in students:
                        
                        print("Recognized =", name)
                        #already_appeared.append(name)
                        students.remove(name)
                        print(students)
                        #time.sleep(3)
                        current_time = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        current_date = datetime.datetime.now().strftime("%m/%d/%Y")
                        lnwriter.writerow([name, current_time, current_date, current_location])
                 
            faces = face_capture.detectMultiScale(colour, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))

            for (x, y, w, h) in faces:
                x=x+55
                y=y+162
                cv2.rectangle(imagebackground, (x, y), (x + w, y + h), (50, 50, 255), 2)

                # Add the name at the lower corner of the rectangle
                # x=x+50
                # y=y+150
                cv2.putText(imagebackground, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            #cv2.imshow('FACE DETECTION...', image_per_sec)
            cv2.imshow('Face Attendance',imagebackground)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_file_path = "students1.csv"
    
    known_face_encoding, known_faces_names = load_known_faces(csv_file_path)
    students = known_faces_names.copy()
    csv_file_name = create_csv_file()
    
    recognize_faces(known_face_encoding, known_faces_names, students, csv_file_name)
