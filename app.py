import cv2
import numpy as np
import time

# Capture video
cap = cv2.VideoCapture(0)

# Optional: save output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = None

time.sleep(3)  # warm-up

# Capture background (wait for empty frame)
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ==============================
    # Black cloak detection
    # ==============================
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

    # Inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment out cloak & background
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    res2 = cv2.bitwise_and(background, background, mask=mask)

    # Final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Initialize VideoWriter when frame size is known
    if out is None:
        out = cv2.VideoWriter("output.avi", fourcc, 20.0,
                              (final_output.shape[1], final_output.shape[0]))

    # Write frame to file
    out.write(final_output)

    # Show window
    cv2.imshow("Harry Potter Vanishing Effect - Black Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
