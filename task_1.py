import cv2 as cv
import numpy as np

def lucas_kanade_optical_flow(prev_gray, new_gray, p0, window_size):
    half_window = window_size // 2
    Ix = cv.Sobel(prev_gray, cv.CV_64F, 1, 0, ksize=3)
    Iy = cv.Sobel(prev_gray, cv.CV_64F, 0, 1, ksize=3)
    It = new_gray.astype(float) - prev_gray.astype(float)

    flow_vectors = []
    h, w = prev_gray.shape  # Get image dimensions

    for point in p0:
        x, y = point.ravel().astype(int)

        # Ensure the window doesn't exceed image boundaries
        if x - half_window < 0 or x + half_window >= w or y - half_window < 0 or y + half_window >= h:
            continue

        Ix_win = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        Iy_win = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        It_win = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

        A = np.vstack((Ix_win, Iy_win)).T  # Shape (N,2)
        b = -It_win

        if A.shape[0] >= 2:
            v = np.linalg.pinv(A) @ b
            flow_vectors.append((x, y, v[0], v[1]))  # Store flow

    return flow_vectors

feature_params = dict(maxCorners=3000, qualityLevel=0.2, minDistance=2, blockSize=7)

cap = cv.VideoCapture("videos/clip_2.mp4")



ret, first_frame = cap.read()

if not ret:
    print("Error: Cannot read video")
    cap.release()
    exit()

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
p0 = np.float32(p0) 

mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    flow_vectors = lucas_kanade_optical_flow(prev_gray, new_gray, p0, 5)
    
    if flow_vectors:
        for x, y, u, v in flow_vectors:
            x_new, y_new = int(x + u), int(y + v)
            mask = cv.line(mask, (x, y), (x_new, y_new), (255, 255, 255), 2)
            frame = cv.circle(frame, (x_new, y_new), 3, (0, 0, 255), -1)

        output = cv.add(frame, mask)
        cv.imshow('Lucas-Kanade Optical Flow (Manual)', output)

        # Update previous points
        p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        p0 = np.float32(p0)
    else:
        print("No motion detected")
        continue  # Continue to next frame if no motion is detected

    # Update previous frame
    prev_gray = new_gray.copy()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()