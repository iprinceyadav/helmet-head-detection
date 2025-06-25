import cv2
from ultralytics import YOLO
import time
import smtplib
from email.message import EmailMessage
import ssl
import csv
from datetime import datetime
import os

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"                           # Your trained YOLO model
LOG_FILE = "helmet_violations_log.csv"           # CSV log file
NOTIFICATION_INTERVAL = 15                       # Minimum seconds between emails

# Email configuration
SENDER_EMAIL = "yadav27prince@gmail.com"            # Replace with your sender email
SENDER_PASSWORD = ""            # Use app password (not your email password)
RECEIVER_EMAIL = "231b231@juetguna.in"     # Replace with manager's email

# --- INITIALIZE ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
last_notification_time = 0

# Initialize CSV if it doesn't exist
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "People Without Helmet", "Image Filename"])

# --- SEND EMAIL FUNCTION ---
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

    # Create email
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(body)

    # Attach image
    with open(image_path, "rb") as img:
        img_data = img.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=image_path)

    # Send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

    print("📧 Email sent to manager.")

# --- MAIN DETECTION LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam not working.")
        break

    results = model(frame)
    names = results[0].names
    boxes = results[0].boxes
    detected_classes = [names[int(cls)] for cls in boxes.cls]

    head_count = detected_classes.count("head")
    helmet_count = detected_classes.count("helmet")
    no_helmet_count = head_count  # assuming heads = no helmet

    print(f"Detected: {detected_classes}")
    print(f"Heads: {head_count}, Helmets: {helmet_count}, No Helmet: {no_helmet_count}")

    current_time = time.time()
    if no_helmet_count > 0 and (current_time - last_notification_time > NOTIFICATION_INTERVAL):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"violation_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)

        # Send email with image
        send_email_alert(no_helmet_count, image_filename)

        # Log to CSV
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, no_helmet_count, image_filename])

        last_notification_time = current_time

    # Display annotated result
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
