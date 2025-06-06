from ultralytics import YOLO
import cv2
import winsound  # For sound alert on Windows

model = YOLO('best.pt')  # Your trained YOLO model
cap = cv2.VideoCapture(0)

# Define colors for specific classes
class_colors = {
    'helmet': (0, 255, 0),  # Green
    'head': (0, 0, 255),    # Red
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    alert_triggered = False  # Flag to avoid multiple alerts in one frame

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = model.names[cls]

            color = class_colors.get(label, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # If 'head' is detected and alert hasn't been triggered yet
            if label == 'head' and not alert_triggered:
                winsound.Beep(1000, 300)  # Beep at 1000 Hz for 300 ms
                alert_triggered = True

    cv2.imshow("Helmet Detection Alert System", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
