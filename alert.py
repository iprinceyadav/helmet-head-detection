import cv2
from ultralytics import YOLO
import time
import requests

# Load the model
model = YOLO("best.pt")

# Telegram credentials
BOT_TOKEN = ""
CHAT_ID = ""

# Initialize webcam
cap = cv2.VideoCapture(0)

# Notification interval
last_notification_time = 0
notification_interval = 15  # seconds

# Send Telegram alert with count
def send_telegram_alert(count):
    message = f"🚨 {count} person(s) without helmet detected! Please take immediate action."
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Telegram alert sent.")
    else:
        print("❌ Failed to send Telegram alert:", response.text)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam error")
        break

    results = model(frame)
    names = results[0].names
    boxes = results[0].boxes
    classes = [names[int(cls)] for cls in boxes.cls]

    # Count detected objects
    head_count = classes.count("head")
    helmet_count = classes.count("helmet")

    # People without helmets = heads detected but not helmets
    no_helmet_count = head_count  # assuming heads = without helmets

    print(f"Detected: {classes}")
    print(f"Heads: {head_count}, Helmets: {helmet_count}, Without Helmet: {no_helmet_count}")

    if no_helmet_count > 0:
        current_time = time.time()
        if current_time - last_notification_time > notification_interval:
            send_telegram_alert(no_helmet_count)
            last_notification_time = current_time

    # Display annotated video
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
















#BOT_TOKEN = "7687509396:AAEeTTwtleCld7lY9TGCzmTrcQkDGnjntdM"  # replace with yours
#CHAT_ID = "1846471425"  # replace with manager's chat ID
