import cv2
import os
import string
import time
signer = input("Enter Signer name:").strip()
labels = list(string.ascii_uppercase) + list("0123456789")
goal = 20         # photos per sign
interval = 2      # seconds between photos
get_ready = 15    # seconds to check the new sign

i = 0
cam = cv2.VideoCapture(0)
while i < len(labels):
    label = labels[i]
    folder = os.path.join("data", signer, label)
    os.makedirs(folder, exist_ok=True)
    count = len(os.listdir(folder))
    next_shot = time.time() + get_ready
    while count < goal:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w // 8, h // 6, 7 * w // 8, 5 * h // 6
        show = frame.copy()
        cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)

        seconds_left = max(0, next_shot - time.time())
        cv2.putText(show, f"Sign: {label}   {count}/{goal}   ({i + 1} of {len(labels)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(show, f"Next photo in {seconds_left:.1f}s   (s = start now, q = quit)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Collect", show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key in (ord('q'), ord('Q')):
            print("Quitting...")
            i = len(labels)          # stops the whole program
            break
        elif key in (ord('s'), ord('S')):
            next_shot = time.time()  # start now

        if time.time() >= next_shot:
            img_name = f"{label}_{count}.png"
            img_path = os.path.join(folder, img_name)
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(img_path, crop)
            print(f"Saved {img_path}")
            count += 1
            next_shot = time.time() + interval
    i += 1
cam.release()
cv2.destroyAllWindows()
print("Finished.")