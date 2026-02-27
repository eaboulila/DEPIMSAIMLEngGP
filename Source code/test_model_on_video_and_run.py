import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  # your trained model
video_path = "road.mp4"  # your video file

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.7)  # adjust threshold 
    annotated = results[0].plot()

    cv2.imshow("YOLO Video Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
