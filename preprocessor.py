import cv2
import numpy as np
import os

def align_with_sift(template_path, photo_path):
    
    """
    Aligns photo with template using SIFT feature matching.
    """
    
    print("Loading images...")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

    if template is None or photo is None:
        raise ValueError("[Error] Could not load images. Check paths.")

    print("Detecting SIFT keypoints...")
    sift = cv2.SIFT_create()

    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_photo, des_photo = sift.detectAndCompute(photo, None)

    print("Matching features...")
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_template, des_photo, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} high quality feature matches.")

    if len(good_matches) > 10:
        print("Calculating homography and warping image...")

        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        color_photo = cv2.imread(photo_path)

        h, w = template.shape
        aligned_img = cv2.warpPerspective(color_photo, matrix, (w, h))

        # cv2.imwrite(output_path, aligned_img)
        # print(f"Aligned image saved to {output_path}")
        return aligned_img
    else:
        raise ValueError(f"Not enough features matched to align the image. Found only {len(good_matches)}.")
    
def balance_lighting_clahe(img):
    """
    Balances the lighting of the raw photo using CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    """

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    shadow_map = cv2.GaussianBlur(l_channel, (151, 151), 0)
    normalised_l = cv2.divide(l_channel, shadow_map, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(normalised_l)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    balanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return balanced_img

def run(template:str, photo: str):
    
    template_path = f"template/{template}.png"
    photo_path = f"raw_photo/{photo}.png"
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"[Error] Template image not found: {template}")
    
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"[Error] File not found: {photo}")
    
    aligned = align_with_sift(template_path, photo_path)
    processed = balance_lighting_clahe(aligned)
    
    output_path = f"processed_image/{photo}_processed.png"
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to {output_path}")

    return output_path