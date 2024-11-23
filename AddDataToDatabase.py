import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://facerecognitionpython-dc12e-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

date = {
    "100000":
        {
            "name": "Sidharth Manikuttan",
            "major": "CSE",
            "starting_year": 2021,
            "total_attendance": 0,
            "Batch": "A",
            "year": 4,
            "last_attendance_time": "2024-11-03 00:54:34"
        },
    "100001":
        {
            "name": "Emly Blunt",
            "major": "Civil",
            "starting_year": 2023,
            "total_attendance": 0,
            "Batch": "B",
            "year": 2,
            "last_attendance_time": "2024-11-03 00:54:34"
        },
    "100002":
            {
                "name": "Elon Musk",
                "major": "Mechanics",
                "starting_year": 2023,
                "total_attendance": 0,
                "Batch": "A",
                "year": 2,
                "last_attendance_time": "2024-11-03 00:54:34"
            },
    "100003":
            {
                "name": "Sreelakshmi",
                "major": "CSE",
                "starting_year": 2021,
                "total_attendance": 0,
                "Batch": "B",
                "year": 4,
                "last_attendance_time": "2024-11-03 00:54:34"
            },
    "100004":
            {
                "name": "Devika K M",
                "major": "CSE",
                "starting_year": 2021,
                "total_attendance": 0,
                "Batch": "B",
                "year": 4,
                "last_attendance_time": "2024-11-03 00:54:34"
            }
}

for key, value in date.items():
    ref.child(key).set(value)
