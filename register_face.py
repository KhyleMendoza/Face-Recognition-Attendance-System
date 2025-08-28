import cv2
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
KNOWN_DIR = APP_DIR / "known"

def main():
	KNOWN_DIR.mkdir(parents=True, exist_ok=True)
	name = input("Enter person's name (used as filename): ").strip()
	if not name:
		print("Invalid name.")
		return
	filename = KNOWN_DIR / f"{name}.jpg"

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Cannot access camera.")
		return

	print("Press SPACE to capture, Q to quit.")
	img = None
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		cv2.imshow("Capture Face", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord(' '):
			img = frame.copy()
			break
		if key in (ord('q'), ord('Q')):
			break

	cap.release()
	cv2.destroyAllWindows()

	if img is not None:
		cv2.imwrite(str(filename), img)
		print(f"Saved {filename}. Now run main.py to recognize.")

if __name__ == "__main__":
	main()