import cv2
import numpy as np

class PCBInspector:
    def __init__(self, reference_path):
        self.ref_img = cv2.imread(reference_path)
        if self.ref_img is None:
            raise ValueError(f"Could not load reference image from {reference_path}")
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.ref_gray.shape

        # Initialize aligner (ORB)
        self.orb = cv2.ORB_create(5000)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.ref_gray, None)
        
    def align_image(self, img):
        """Aligns the input image to the reference image using homography."""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(img_gray, None)

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if des2 is None:
            print("No features found in test image.")
            return img, None
            
        matches = bf.match(self.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep top matches
        clean_matches = matches[:int(len(matches) * 0.15)]
        
        if len(clean_matches) < 4:
            print("Not enough matches for homography.")
            return img, None

        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in clean_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in clean_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print("Homography could not be computed.")
            return img, None
            
        aligned_img = cv2.warpPerspective(img, M, (self.width, self.height))
        return aligned_img, M

    def inspect(self, test_img_path):
        """Main inspection loop for a single image."""
        test_img = cv2.imread(test_img_path)
        if test_img is None:
            print(f"Failed to load {test_img_path}")
            return None

        aligned, M = self.align_image(test_img)
        if aligned is None:
            return None

        # Difference
        diff = cv2.absdiff(self.ref_img, aligned)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Cleanup noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: # filter small noise
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extract ROI for classification
            roi_ref = self.ref_img[y:y+h, x:x+w]
            roi_test = aligned[y:y+h, x:x+w]
            
            defect_type, confidence = self.classify_defect(roi_ref, roi_test, (w, h))
            
            results.append({
                "type": defect_type,
                "confidence": confidence,
                "bbox": (x, y, w, h),
                "center": (x + w//2, y + h//2)
            })
            
        return aligned, results

    def classify_defect(self, roi_ref, roi_test, size):
        w, h = size
        aspect_ratio = float(w)/h if h > 0 else 0
        
        # Diff in HSV
        hsv_ref = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2HSV)
        hsv_test = cv2.cvtColor(roi_test, cv2.COLOR_BGR2HSV)
        
        # Calculate mean differences
        diff_h = np.abs(np.mean(hsv_ref[:,:,0]) - np.mean(hsv_test[:,:,0]))
        diff_s = np.abs(np.mean(hsv_ref[:,:,1]) - np.mean(hsv_test[:,:,1]))
        diff_v = np.abs(np.mean(hsv_ref[:,:,2]) - np.mean(hsv_test[:,:,2]))
        
        # Heuristics
        # Scratch: usually bright lines on dark background or vice versa, structural
        # Discoloration: High Hue difference, but structure remains
        # Missing Component: Large pixel difference, V channel drops (if board is dark)
        
        if diff_h > 20: 
            return "Discoloration", 0.85
        
        if (aspect_ratio > 3 or aspect_ratio < 0.33):
            return "Scratch", 0.90
            
        # Check if missing component (assumes dark board background)
        # If test ROI is significantly darker or greener (if green board) than ref roi (component)
        # Simple check: Mean V difference is large?
        if diff_v > 40:
             return "Missing Component", 0.88

        # Fallback
        return "Defect", 0.50
