import cv2
from ultralytics import YOLO
import time
import smtplib
from email.message import EmailMessage
import csv
from datetime import datetime
import os
import ssl
import getpass

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
LOG_FILE = "helmet_violations_log.csv"
NOTIFICATION_INTERVAL = 15  # seconds

# Email setup
SMTP_SERVER = "smtp.gmail.com"  # Or "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "yadav27prince12@gmail.com"
RECEIVER_EMAIL = "anishkumar.gupta@adityabirla.com"
SENDER_PASSWORD = "woamsstkhytmrnme"  # Secure prompt

# --- Initialize YOLO Model ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
last_notification_time = 0

# Initialize log file
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "People Without Helmet", "Image Filename"])

# --- Send Email Alert ---
def send_email_alert(count, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = "🚨 Helmet Violation Alert"
    body = f"""
Hello,

{count} person(s) were detected without a helmet.

📅 Time: {timestamp}
📷 See attached image.

Regards,
Safety Monitoring System
"""

    # Build the email message
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

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

# --- Detection Loop ---
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
            if float(conf) > 0.5:  # Confidence threshold
                detected_classes.append(names[int(cls_id)])

    head_count = detected_classes.count("head")
    helmet_count = detected_classes.count("helmet")
    no_helmet_count = head_count - helmet_count
    no_helmet_count = max(no_helmet_count, 0)

    print(f"Detected: {detected_classes}")
    print(f"Heads: {head_count}, Helmets: {helmet_count}, No Helmet: {no_helmet_count}")

    current_time = time.time()
    if no_helmet_count > 0 and (current_time - last_notification_time > NOTIFICATION_INTERVAL):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("violations", exist_ok=True)
        image_filename = f"violations/violation_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)

        send_email_alert(no_helmet_count, image_filename)

        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, no_helmet_count, image_filename])

        last_notification_time = current_time

    # Show annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
