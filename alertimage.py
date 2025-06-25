import cv2
from ultralytics import YOLO
import time
import requests

# Load the model
model = YOLO("best.pt")

# Telegram credentials
BOT_TOKEN = "7687509396:AAEeTTwtleCld7lY9TGCzmTrcQkDGnjntdM"         # replace with your bot token
CHAT_ID = "1846471425"             # replace with your manager's chat ID

# Webcam setup
cap = cv2.VideoCapture(0)

# Notification interval (to avoid spamming)
last_notification_time = 0
notification_interval = 15  # seconds

# Function to send Telegram photo + message
def send_telegram_alert(count, frame):
    message = f"🚨 {count} person(s) without helmet detected! Please take immediate action."

    # Save frame temporarily
    filename = "violation.jpg"
    cv2.imwrite(filename, frame)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(filename, "rb") as photo:
        payload = {
            "chat_id": CHAT_ID,
            "caption": message
        }
        files = {
            "photo": photo
        }
        response = requests.post(url, data=payload, files=files)

    if response.status_code == 200:
        print("📷 Telegram image and alert sent successfully.")
    else:
        print("❌ Failed to send photo:", response.text)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error reading webcam")
        break

    results = model(frame)
    names = results[0].names
    boxes = results[0].boxes
    detected_classes = [names[int(cls)] for cls in boxes.cls]

    # Count heads and helmets
    head_count = detected_classes.count("head")
    helmet_count = detected_classes.count("helmet")

    # Logic: assume head = without helmet
    no_helmet_count = head_count

    print(f"Detected: {detected_classes}")
    print(f"Heads: {head_count}, Helmets: {helmet_count}, Without Helmet: {no_helmet_count}")

    # If any person without helmet
    if no_helmet_count > 0:
        current_time = time.time()
        if current_time - last_notification_time > notification_interval:
            send_telegram_alert(no_helmet_count, frame)
            last_notification_time = current_time

    # Display output
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
