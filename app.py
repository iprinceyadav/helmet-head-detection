from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import winsound

app = Flask(__name__)
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

class_colors = {
    'helmet': (0, 255, 0),
    'head': (0, 0, 255),
}

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO inference
        results = model(frame, stream=True)
        alert_triggered = False

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

                if label == 'head' and not alert_triggered:
                    winsound.Beep(1000, 300)
                    alert_triggered = True

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
