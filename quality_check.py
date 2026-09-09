import cv2
import numpy as np
import os

def check_image_quality(template_path, photo_path, blur_threshold=100.0, glare_threshold=5.0, dark_threshold=40.0, quad_threshold=0.40):
    """checks the quality of the processed image against the template image based on optical and geometric criteria."""
    
    print(f"--- Analysing Quality for: {os.path.basename(photo_path)} ---")
    
    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if photo is None or template is None:
        raise ValueError("[Error] Could not load images. Check your paths.")

    passed = True
    print("\n[1/2] Running Optical Checks...")

    blur_score = cv2.Laplacian(photo, cv2.CV_64F).var()
    is_blurry = blur_score < blur_threshold
    print(f"  - Blur Score:       {blur_score:.2f} (Needs > {blur_threshold})")

    # Initial check for partial glare but will will be commented out for now
    # as it catches too many false positives.
    
    # glare_pixels = np.sum(photo > 240)
    # glare_percentage = (glare_pixels / photo.size) * 100
    # has_glare = glare_percentage > glare_threshold
    # print(f"  - Glare Percentage: {glare_percentage:.2f}% (Needs < {glare_threshold}%)")

    mean_brightness = np.mean(photo)
    is_dark = mean_brightness < dark_threshold
    print(f"  - Mean Brightness:  {mean_brightness:.2f} (Needs > {dark_threshold})")

    print("\n[2/2] Running Geometric Quadrant Analysis (Folds & Angles)...")
    
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_photo, des_photo = sift.detectAndCompute(photo, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_template, des_photo, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    has_bad_geometry = False

    if len(good_matches) < 20:
        print("Geometric Check: FAILED (Not enough matches found).")
        has_bad_geometry = True
    else:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        h, w = template.shape
        mid_x, mid_y = w // 2, h // 2

        quadrants = {
            "Top-Left": [0, 0],
            "Top-Right": [0, 0],
            "Bottom-Left": [0, 0],
            "Bottom-Right": [0, 0]
        }

        for i, match in enumerate(good_matches):
            pt = kp_template[match.queryIdx].pt
            x, y = pt[0], pt[1]
            is_inlier = mask[i][0] == 1

            if x < mid_x and y < mid_y:
                q = "Top-Left"
            elif x >= mid_x and y < mid_y:
                q = "Top-Right"
            elif x < mid_x and y >= mid_y:
                q = "Bottom-Left"
            else:
                q = "Bottom-Right"

            quadrants[q][0] += 1
            if is_inlier:
                quadrants[q][1] += 1

        min_ratio = 1.0
        worst_quadrant = ""

        for q_name, (total, inliers) in quadrants.items():
            if total < 5:
                print(f"  - {q_name}: FAILED (Virtually no features detected)")
                ratio = 0.0
            else:
                ratio = inliers / total
                status = "PASS" if ratio >= quad_threshold else "FAIL"
                print(f"  - {q_name}: {ratio:.2%} inliers ({status})")
            
            if ratio < min_ratio:
                min_ratio = ratio
                worst_quadrant = q_name

        if min_ratio < quad_threshold:
            has_bad_geometry = True
            print(f"Geometry Check FAILED. The {worst_quadrant} is severely distorted/folded.")

    print("\n--- Final Decision ---")
    
    if is_blurry:
        print("REJECTED: Image is too blurry. Please retake and hold the camera steady.")
        passed = False
    if is_dark:
        print("REJECTED: Image is too dark. Please turn on a light.")
        passed = False
    if has_bad_geometry:
        print("REJECTED: Severe folds or bad camera angle detected. Please flatten the paper and shoot straight down.")
        passed = False
        
    if passed:
        print("ACCEPTED: No issue detected.")
        
    return passed

def run(template:str, photo: str):
    
    template_path = f"template/{template}.png"
    photo_path = f"processed_image/{photo}_processed.png"
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"[Error] Template image not found: {template}")
    
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"[Error] File not found: {photo}")
    
    check_image_quality(template_path, photo_path)