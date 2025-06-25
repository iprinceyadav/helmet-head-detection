import cv2
from ultralytics import YOLO
import time
import smtplib
from email.message import EmailMessage
import csv
from datetime import datetime
import os
import ssl

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
LOG_FILE = "helmet_violations_log.csv"
NOTIFICATION_INTERVAL = 15  # seconds

# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "yadav27prince12@gmail.com"
RECEIVER_EMAIL = "231b231@juetguna.in"
SENDER_PASSWORD = "woamsstkhytmrnme"  # Use an App Password for Gmail

# --- Initialize YOLO Model ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
last_notification_time = 0

# Initialize log file if not exists
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "People Without Helmet", "Video Filename"])

# --- Function to record a 5-second video clip ---
def record_violation_clip(cap, filename, duration=5, fps=20):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        # Optional: display recording
        cv2.imshow("Recording Violation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyWindow("Recording Violation")

# --- Function to send email alert with video attachment ---
def send_email_alert(count, video_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = "🚨 Helmet Violation Alert"
    body = f"""
Hello,

{count} person(s) were detected without a helmet.

📅 Time: {timestamp}
📹 See attached 5-second clip.

Regards,
Safety Monitoring System
"""

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(body)

    # Attach the video file
    with open(video_path, "rb") as f:
        video_data = f.read()
        msg.add_attachment(video_data, maintype="video", subtype="mp4", filename=os.path.basename(video_path))

    try:
        smtp = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.send_message(msg)
        smtp.quit()
        print("📧 Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- Main Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to access webcam.")
        break

    results = model(frame)
    names = model.names
    boxes = results[0].boxes

    detected_classes = []
    if boxes is not None and boxes.cls is not None:
        for cls_id, conf in zip(boxes.cls, boxes.conf):
            if float(conf) > 0.5:
                detected_classes.append(names[int(cls_id)])

    head_count = detected_classes.count("head")
    helmet_count = detected_classes.count("helmet")
    no_helmet_count = max(head_count - helmet_count, 0)

    print(f"Detected: {detected_classes}")
    print(f"Heads: {head_count}, Helmets: {helmet_count}, No Helmet: {no_helmet_count}")

    current_time = time.time()
    if no_helmet_count > 0 and (current_time - last_notification_time > NOTIFICATION_INTERVAL):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("violations", exist_ok=True)
        video_filename = f"violations/violation_{timestamp}.mp4"

        # Record 5-second video
        print("📹 Recording 5-second video...")
        record_violation_clip(cap, video_filename, duration=5)

        # Send email alert with video
        send_email_alert(no_helmet_count, video_filename)

        # Log the violation
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, no_helmet_count, video_filename])

        last_notification_time = current_time

    # Show live detection
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
