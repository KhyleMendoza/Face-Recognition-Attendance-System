import os
import csv
import cv2
import time
import datetime as dt
import numpy as np
import face_recognition
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
KNOWN_DIR = APP_DIR / "known"
DATA_DIR = APP_DIR / "data"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

MODEL = "hog"
TOLERANCE = 0.5
FRAME_RESIZE = 0.25

def ensure_dirs():
	KNOWN_DIR.mkdir(parents=True, exist_ok=True)
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	if not ATTENDANCE_CSV.exists():
		with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["date", "time", "name"])

def load_known_faces():
	encodings = []
	names = []
	for p in sorted(KNOWN_DIR.iterdir()):
		if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
			continue
		image = face_recognition.load_image_file(str(p))
		boxes = face_recognition.face_locations(image, model=MODEL)
		if not boxes:
			print(f"[WARN] No face found in {p.name}; skipping.")
			continue
		encs = face_recognition.face_encodings(image, boxes)
		if not encs:
			print(f"[WARN] No encodings for {p.name}; skipping.")
			continue
		name = p.stem
		encodings.append(encs[0])
		names.append(name)
	print(f"[INFO] Loaded {len(encodings)} known faces.")
	return encodings, names

def read_today_names():
	today = dt.date.today().isoformat()
	seen = set()
	if ATTENDANCE_CSV.exists():
		with ATTENDANCE_CSV.open("r", encoding="utf-8") as f:
			r = csv.DictReader(f)
			for row in r:
				if row["date"] == today:
					seen.add(row["name"])
	return seen

def mark_attendance(name):
	now = dt.datetime.now()
	row = [now.date().isoformat(), now.strftime("%H:%M:%S"), name]
	with ATTENDANCE_CSV.open("a", newline="", encoding="utf-8") as f:
		cw = csv.writer(f)
		cw.writerow(row)
	print(f"[ATTENDANCE] {row}")

def main():
	ensure_dirs()
	known_encodings, known_names = load_known_faces()
	if not known_encodings:
		print("[ERROR] No known faces found in 'known/'. Add images named as the person's name.")
		return

	seen_today = read_today_names()
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("[ERROR] Cannot access camera.")
		return

	prev = time.time()
	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				print("[WARN] Frame grab failed.")
				break

			now = time.time()
			fps = 1.0 / (now - prev) if now > prev else 0.0
			prev = now

			small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
			rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

			locations = face_recognition.face_locations(rgb_small, model=MODEL)
			encodings = face_recognition.face_encodings(rgb_small, locations)

			names_in_frame = []
			for enc in encodings:
				distances = face_recognition.face_distance(known_encodings, enc)
				if len(distances) == 0:
					names_in_frame.append("Unknown")
					continue
				best_idx = np.argmin(distances)
				if distances[best_idx] <= TOLERANCE:
					names_in_frame.append(known_names[best_idx])
				else:
					names_in_frame.append("Unknown")

			for (top, right, bottom, left), name in zip(locations, names_in_frame):
				top = int(top / FRAME_RESIZE)
				right = int(right / FRAME_RESIZE)
				bottom = int(bottom / FRAME_RESIZE)
				left = int(left / FRAME_RESIZE)

				color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
				cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
				cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
				cv2.putText(frame, name, (left + 4, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

				if name != "Unknown" and name not in seen_today:
					mark_attendance(name)
					seen_today.add(name)

			cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
			cv2.putText(frame, "Press Q to quit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

			cv2.imshow("Face Recognition Attendance", frame)
			if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	main()