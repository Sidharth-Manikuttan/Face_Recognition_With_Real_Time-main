import os
import pickle
import cvzone
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "sidharthmanikuttan9@gmail.com"  
SENDER_APP_PASSWORD = "lfvp adna dopi pvsy"  
RECEIVER_EMAIL = "notifysidharth@gmail.com" 

def send_email(person_name):
    subject = "Person Detected by SMART GUARD"
    body = f"{person_name} has been detected by your face recognition security system."
    
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(message)
        print("Email notification sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://facerecognitionpython-dc12e-default-rtdb.firebaseio.com/",
   'storageBucket': "facerecognitionpython-dc12e.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# importing modes images
folderModePath = 'Resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# loading the encoding file
print("loading the encoded file...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("encoded file loaded...")

modeType = 0
counter = 0
id = -1
imgStudent = []
email_sent = False  # Flag to track if email has been sent

while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                bbox = 55+x1, 162+y1, x2-x1, y2-y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]

                # Retrieve the student's name from the Firebase Realtime Database
                student_ref = db.reference(f'Students/{id}')
                student_data = student_ref.get()
                person_name = student_data['name']

                # Send email if not sent already
                if not email_sent:
                    threading.Thread(target=send_email, args=(person_name,)).start()
                    email_sent = True
            else:
                # Unknown face detected - show Mode 4 (Stranger)
                modeType = 4
                counter = 0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                continue  # Skip the rest of the loop for unknown faces

        if counter == 0:
                    cvzone.putTextRect(imgBackground,"Loading", (275,400))
                    cv2.imshow("Face Attendance",imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                #get image from storage
                blob = bucket.get_blob(f'images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGR2RGB)
                imgStudent = cv2.resize(imgStudent, (216, 216))
                #update attendences
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    
                    # Show Mode 1 (student info) for longer, then Mode 2 (Marked)
                    modeType = 1
                else:
                    # If already marked within 30 seconds, go directly to Mode 3 (Already Marked)
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            # Modified the mode transitions
            if modeType == 1:
                if counter <= 25:  # Extended time for Mode 1 (student info)
                    modeType = 1
                elif counter <= 35:  # Show Mode 2 (Marked) for a shorter time
                    modeType = 2
                else:  # Finally show Mode 3 (Already Marked)
                    modeType = 3

            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
            
            if modeType != 3:
                if counter <= 25:  # Show student info during Mode 1
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(imgBackground, str(studentInfo['Batch']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414-w)//2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackground[175:175+216, 909:909+216] = imgStudent

                counter += 1

                if counter >= 35:  # After Mode 2, transition to Mode 3
                    counter = 0
                    studentInfo = []
                    imgStudent = []
                    modeType = 3
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    email_sent = False  # Reset the email sent flag
    else:
        modeType = 0
        counter = 0
        email_sent = False  # Reset the email sent flag
    
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)