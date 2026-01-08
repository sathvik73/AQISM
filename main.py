import os
import cv2
import glob
import json
from src.inspector import PCBInspector

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    output_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ref_path = os.path.join(dataset_dir, 'golden_master.png')
    
    print("Initializing Inspector...")
    try:
        inspector = PCBInspector(ref_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Process all test images
    test_images = glob.glob(os.path.join(dataset_dir, 'test_*.png'))
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        print(f"\nProcessing {filename}...")
        
        aligned_img, results = inspector.inspect(img_path)
        
        if aligned_img is None:
            print("Failed to process image.")
            continue
            
        # Draw results
        output_img = aligned_img.copy()
        
        if not results:
            print("No defects detected.")
            status_text = "PASS"
            color = (0, 255, 0)
        else:
            print(f"Found {len(results)} defects:")
            status_text = "FAIL"
            color = (0, 0, 255)
            
            for res in results:
                x, y, w, h = res['bbox']
                dtype = res['type']
                conf = res['confidence']
                print(f" - {dtype} at ({x}, {y}) conf: {conf:.2f}")
                
                # Draw box
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # Draw label
                label = f"{dtype} ({conf:.2f})"
                cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw overall status
        cv2.putText(output_img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Save output
        out_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(out_path, output_img)
        print(f"Saved result to {out_path}")

if __name__ == "__main__":
    main()
