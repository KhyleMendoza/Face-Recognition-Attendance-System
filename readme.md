## Face Recognition Attendance System

Real-time attendance logging using OpenCV and `face_recognition`. Loads face encodings from `known/` (image filename = person name), recognizes via webcam, and logs first check-in per day to `data/attendance.csv`.

### Install Instruction
pip install -r requirements.txt

### Run Project
python main.py

### Add faces Capture from webcam:
python register_face.py

- Enter the person’s name when prompted (used as the filename).
- Press SPACE to capture a frame, `Q` to quit.
- An image like `known/John Doe.jpg` will be saved.
