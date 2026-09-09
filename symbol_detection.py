import easyocr
import cv2
import os
import json
import argparse
import torch
from torchvision import transforms, models
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using hardware accelerator: {device}")
with open('class_mapping.json', 'r') as f:
    class_names = json.load(f)

classification_model = models.mobilenet_v2(weights=None)
classification_model.classifier[1] = torch.nn.Linear(classification_model.classifier[1].in_features, len(class_names))
classification_model.load_state_dict(torch.load('symbol_classifier.pth', map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()

infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser(description="OCR engine: easyOCR")
    parser.add_argument("-i", "--image", required=True, help="Path to the image.")
    parser.add_argument("-j", "--json", required=True, help="Path to the COCO JSON file.")
    parser.add_argument("-c", "--column", help="Column name to run ocr.")
    
    args = parser.parse_args()

    if not os.path.exists(args.image) or not os.path.exists(args.json):
        print("[Error] Image or JSON file not found.")
        return

    img = cv2.imread(args.image)
    with open(args.json, 'r') as f:
        data = json.load(f)

    if 'annotations' not in data:
        print("[Error] JSON does not contain 'annotations' array.")
        return
    
    results = ocr(args.column, img, data, offset_y=0)
    
def predict_symbol(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    input_tensor = infer_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classification_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

def ocr(column, img, data, offset_y=0):
    print("Initialising EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    print("\n--- OCR Results ---")

    text_results = []

    for ann in data.get('annotations', []):
        attributes = ann.get('attributes', {})

        if attributes.get('column') == column:
            row_num = attributes.get('row')
            x, y, w, h = ann['bbox']
            pad = 0
            
            x1 = int(x) + pad
            y1 = int(y) + offset_y + pad
            x2 = int(x + w) - pad
            y2 = int(y + h) + offset_y - pad

            if x1 >= x2 or y1 >= y2 or y1 < 0 or y2 > img.shape[0]:
                print(f"Row {row_num}: [Crop Out of Bounds]")
                continue

            roi = img[y1:y2, x1:x2]

            scale_factor = 3
            upscaled = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            blur = cv2.blur(upscaled, (5, 5))

            allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'

            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/debug_row_{row_num}.png", blur)
            
            ocr_result = reader.readtext(blur,
                                         detail=1, 
                                         allowlist=allowed_chars,
                                         text_threshold=0.3,
                                         low_text=0.3)

            texts = []
            confidences = []
            for bbox, text, conf in ocr_result:
                texts.append(text)
                confidences.append(conf)
            
            extracted_text = " ".join(texts).strip()
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            
            final_text = extracted_text
            
            suspicious_chars = ["X", "2", "7", "V", "v", ".", "1", "I", "?"]
            is_suspicious_exact_match = extracted_text in suspicious_chars
            
            is_short_and_shaky = len(extracted_text) <= 2 and avg_confidence < 0.6
            
            is_low_confidence = avg_confidence < 0.2 or extracted_text == ""

            if is_low_confidence or is_short_and_shaky or is_suspicious_exact_match:
                final_text = predict_symbol(roi)

            print(f"Row {row_num:02d}: '{final_text}' | Conf: {avg_confidence:.4f} (Raw OCR: '{extracted_text}')")
            
            text_results.append({
                'row': row_num,
                'text': final_text,
                'confidence': avg_confidence,
                'raw_ocr': extracted_text
            })

    return text_results

if __name__ == "__main__":
    main()