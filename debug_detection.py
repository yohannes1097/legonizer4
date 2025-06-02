#!/usr/bin/env python3
"""
Script untuk debug deteksi LEGO objects
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from preprocessing.detector import LegoDetector

def debug_detection_steps(image_path: str):
    """
    Debug setiap step dalam proses deteksi
    """
    print(f"=== Debugging detection untuk: {image_path} ===")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Tidak dapat membaca gambar {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Initialize detector
    detector = LegoDetector(min_contour_area=500)  # Lower threshold for testing
    
    # Step 1: Shadow removal
    print("\n1. Shadow removal...")
    no_shadow = detector.remove_shadows(image)
    cv2.imwrite("debug_1_no_shadow.jpg", no_shadow)
    print("   Saved: debug_1_no_shadow.jpg")
    
    # Step 2: Contrast enhancement
    print("\n2. Contrast enhancement...")
    enhanced = detector.enhance_contrast(no_shadow)
    cv2.imwrite("debug_2_enhanced.jpg", enhanced)
    print("   Saved: debug_2_enhanced.jpg")
    
    # Step 3: Color segmentation
    print("\n3. Color segmentation...")
    color_mask = detector.color_segmentation(enhanced)
    cv2.imwrite("debug_3_color_mask.jpg", color_mask)
    print("   Saved: debug_3_color_mask.jpg")
    print(f"   Color mask non-zero pixels: {np.count_nonzero(color_mask)}")
    
    # Step 4: Grayscale and denoising
    print("\n4. Grayscale and denoising...")
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite("debug_4_denoised.jpg", denoised)
    print("   Saved: debug_4_denoised.jpg")
    
    # Step 5: Adaptive thresholding
    print("\n5. Adaptive thresholding...")
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    cv2.imwrite("debug_5_thresh.jpg", thresh)
    print("   Saved: debug_5_thresh.jpg")
    print(f"   Threshold non-zero pixels: {np.count_nonzero(thresh)}")
    
    # Step 6: Edge detection
    print("\n6. Edge detection...")
    edges = cv2.Canny(denoised, 50, 150)
    cv2.imwrite("debug_6_edges.jpg", edges)
    print("   Saved: debug_6_edges.jpg")
    print(f"   Edges non-zero pixels: {np.count_nonzero(edges)}")
    
    # Step 7: Combine all
    print("\n7. Combining masks...")
    combined = cv2.bitwise_or(thresh, edges)
    combined = cv2.bitwise_and(combined, color_mask)
    cv2.imwrite("debug_7_combined.jpg", combined)
    print("   Saved: debug_7_combined.jpg")
    print(f"   Combined non-zero pixels: {np.count_nonzero(combined)}")
    
    # Step 8: Find contours
    print("\n8. Finding contours...")
    contours, _ = cv2.findContours(
        combined, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"   Total contours found: {len(contours)}")
    
    # Analyze each contour
    valid_contours = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        print(f"   Contour {i}: area = {area}")
        
        if area > detector.min_contour_area:
            print(f"     -> Area check passed")
            if detector.validate_shape(cnt):
                print(f"     -> Shape validation passed")
                valid_contours.append(cnt)
            else:
                print(f"     -> Shape validation FAILED")
        else:
            print(f"     -> Area too small (min: {detector.min_contour_area})")
    
    print(f"\n   Valid contours after filtering: {len(valid_contours)}")
    
    # Draw all contours for visualization
    vis_image = image.copy()
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_8_all_contours.jpg", vis_image)
    print("   Saved: debug_8_all_contours.jpg")
    
    # Draw valid contours
    if valid_contours:
        vis_valid = image.copy()
        cv2.drawContours(vis_valid, valid_contours, -1, (0, 0, 255), 3)
        cv2.imwrite("debug_9_valid_contours.jpg", vis_valid)
        print("   Saved: debug_9_valid_contours.jpg")
    
    return len(valid_contours)

def test_simple_detection(image_path: str):
    """
    Test deteksi sederhana tanpa filter yang kompleks
    """
    print(f"\n=== Testing simple detection untuk: {image_path} ===")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Tidak dapat membaca gambar {image_path}")
        return
    
    # Simple preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Simple method found {len(contours)} contours")
    
    # Filter by area only
    min_area = 500
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Contours with area > {min_area}: {len(large_contours)}")
    
    # Visualize
    vis_simple = image.copy()
    cv2.drawContours(vis_simple, large_contours, -1, (255, 0, 0), 2)
    cv2.imwrite("debug_simple_detection.jpg", vis_simple)
    print("Saved: debug_simple_detection.jpg")
    
    return len(large_contours)

def main():
    # Check if there are any images in data/raw
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print("Creating data/raw directory...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some LEGO images to data/raw/ directory")
        return
    
    # Find image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(raw_dir.glob(f"**/*{ext}")))
        image_files.extend(list(raw_dir.glob(f"**/*{ext.upper()}")))
    
    if not image_files:
        print("No image files found in data/raw/")
        print("Please add some LEGO images to test")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Test first image
    test_image = str(image_files[0])
    print(f"Testing with: {test_image}")
    
    # Debug advanced detection
    advanced_count = debug_detection_steps(test_image)
    
    # Test simple detection
    simple_count = test_simple_detection(test_image)
    
    print(f"\n=== SUMMARY ===")
    print(f"Advanced detection: {advanced_count} objects")
    print(f"Simple detection: {simple_count} objects")
    
    if advanced_count == 0 and simple_count > 0:
        print("Issue: Advanced detection is too restrictive")
    elif simple_count == 0:
        print("Issue: No objects detected even with simple method")
        print("Check if image contains clear LEGO objects")

if __name__ == "__main__":
    main()
