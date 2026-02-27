import cv2
from ultralytics import YOLO

# Load the model 
model = YOLO("best.pt")  

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

print("Webcam running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)
    results = model.predict(frame, conf=0.7)   # adjust threshold 

    # Draw detections
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("YOLO Webcam", annotated_frame)

    # Quit when pressing Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
