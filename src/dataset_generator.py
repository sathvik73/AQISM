import cv2
import numpy as np
import os
import random

def create_scratch(img):
    out = img.copy()
    h, w, _ = out.shape
    # Draw a few random lines
    # Start near center to likely hit something
    for _ in range(3):
        x1 = random.randint(int(w*0.2), int(w*0.8))
        y1 = random.randint(int(h*0.2), int(h*0.8))
        x2 = x1 + random.randint(-50, 50)
        y2 = y1 + random.randint(-50, 50)
        cv2.line(out, (x1, y1), (x2, y2), (220, 220, 220), 4) # Light grey scratch
    return out

def create_missing(img):
    out = img.copy()
    h, w, _ = out.shape
    # Draw a box of the 'board color' over a likely component spot
    # For a real implementation we'd detect components first, but random is okay for prototype
    for _ in range(2):
        start_x = random.randint(100, w-100)
        start_y = random.randint(100, h-100)
        w_rect = random.randint(30, 60)
        h_rect = random.randint(30, 60)
        # Fill with a dark green color similar to the board background
        cv2.rectangle(out, (start_x, start_y), (start_x+w_rect, start_y+h_rect), (20, 50, 25), -1)
    return out

def create_discolor(img):
    out = img.copy()
    h, w, _ = out.shape
    roi_x = random.randint(100, w-200)
    roi_y = random.randint(100, h-200)
    
    roi = out[roi_y:roi_y+100, roi_x:roi_x+100]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Change hue significantly to represent burn/discoloration
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 40) % 180 
    # Also darken it
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(int) - 50, 0, 255)
    
    out[roi_y:roi_y+100, roi_x:roi_x+100] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out

def main():
    base_path = os.path.join(os.path.dirname(__file__), '../dataset/golden_master.png')
    base_path = os.path.abspath(base_path)
    print(f"Loading from: {base_path}")
    
    img = cv2.imread(base_path)
    if img is None:
        print(f"Failed to load {base_path}. Check if file exists.")
        return

    # Scratch
    scratch_img = create_scratch(img)
    scratch_path = os.path.join(os.path.dirname(__file__), '../dataset/test_scratch.png')
    cv2.imwrite(scratch_path, scratch_img)
    print(f"Generated: {scratch_path}")

    # Missing
    missing_img = create_missing(img)
    missing_path = os.path.join(os.path.dirname(__file__), '../dataset/test_missing.png')
    cv2.imwrite(missing_path, missing_img)
    print(f"Generated: {missing_path}")

    # Discolor
    discolor_img = create_discolor(img)
    discolor_path = os.path.join(os.path.dirname(__file__), '../dataset/test_discolor.png')
    cv2.imwrite(discolor_path, discolor_img)
    print(f"Generated: {discolor_path}")
    
    # Clean (Shifted) - for testing alignment
    rows, cols, _ = img.shape
    # Shift by 2 pixels x and y
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    clean_img = cv2.warpAffine(img, M, (cols, rows))
    clean_path = os.path.join(os.path.dirname(__file__), '../dataset/test_clean.png')
    cv2.imwrite(clean_path, clean_img)
    print(f"Generated: {clean_path}")

if __name__ == "__main__":
    main()
