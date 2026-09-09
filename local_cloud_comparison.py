import easyocr
import cv2
import os
import json
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import csv
from google.cloud import vision
from dotenv import load_dotenv

load_dotenv()
GCV_KEY_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

if not GCV_KEY_PATH or not os.path.exists(GCV_KEY_PATH):
    print("[Error] GOOGLE_APPLICATION_CREDENTIALS not found or invalid path in .env file.")
    exit(1)

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

vision_client = vision.ImageAnnotatorClient()

def predict_symbol(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = infer_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classification_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def dual_ocr_extraction(column, img, data, offset_y=0):
    print("Initialising EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    print("\n--- Running OCR Extraction & Classification ---")
    results = []
    os.makedirs("debug", exist_ok=True)

    for ann in data.get('annotations', []):
        attributes = ann.get('attributes', {})

        if attributes.get('column') == column:
            row_num = attributes.get('row')
            x, y, w, h = ann['bbox']
            
            x1, y1 = int(x), int(y) + offset_y
            x2, y2 = int(x + w), int(y + h) + offset_y

            if x1 >= x2 or y1 >= y2 or y1 < 0 or y2 > img.shape[0]:
                print(f"Row {row_num}: [Crop Out of Bounds]")
                continue

            roi = img[y1:y2, x1:x2]
            scale_factor = 3
            upscaled = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            blur = cv2.blur(upscaled, (5, 5))
            cv2.imwrite(f"debug/debug_row_{row_num}.png", blur)
            
            allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'
            ocr_result = reader.readtext(blur, detail=1, allowlist=allowed_chars, text_threshold=0.3, low_text=0.3)
            
            easy_texts = [text for bbox, text, conf in ocr_result]
            easy_confidences = [conf for bbox, text, conf in ocr_result]
            
            easyocr_text = " ".join(easy_texts).strip()
            avg_confidence = sum(easy_confidences) / len(easy_confidences) if easy_confidences else 0.0

            success, encoded_image = cv2.imencode('.png', upscaled)
            content = encoded_image.tobytes()
            vision_image = vision.Image(content=content)
            
            gcv_response = vision_client.document_text_detection(image=vision_image)
            
            if gcv_response.full_text_annotation:
                cloud_text = gcv_response.full_text_annotation.text.strip().replace('\n', ' ')
            else:
                cloud_text = ""

            suspicious_chars = ["X", "2", "7", "V", "v", ".", "1", "I", "?", ""]
            is_suspicious_exact_match = easyocr_text in suspicious_chars
            is_short_and_shaky = len(easyocr_text) <= 2 and avg_confidence < 0.6
            is_low_confidence = avg_confidence < 0.2 or easyocr_text == ""

            is_symbol_cell = is_low_confidence or is_short_and_shaky or is_suspicious_exact_match

            detected_symbol = None
            if is_symbol_cell:
                detected_symbol = predict_symbol(roi)
                cell_type = detected_symbol
            else:
                cell_type = "text"

            print(f"Row {row_num:02d} | EasyOCR: '{easyocr_text:<15}' | GCV: '{cloud_text:<15}' | Detected Type: '{cell_type}'")
            
            results.append({
                'row': row_num,
                'easyocr_raw': easyocr_text,
                'easyocr_conf': avg_confidence,
                'cloud_vision_raw': cloud_text,
                'detected_type': cell_type,
                'is_symbol': is_symbol_cell
            })

    return results

def resolve_and_save_pipeline_results(processed_data, output_name="final_results"):
    print("\n--- Resolving Final Values & Exporting ---")
    
    symbol_map = {
        "hyphen": "N/A",
        "check_mark": "PASS",
        "question_mark": "N/V",
        "blank": ""
    }

    sorted_data = sorted(processed_data, key=lambda x: x.get('row', 0))
    resolved_records = []
    last_valid_text = ""

    for item in sorted_data:
        row_num = item.get('row')
        cell_type = item.get('detected_type', 'text').lower().strip()
        cloud_text = item.get('cloud_vision_raw', '')
        
        final_value = ""
        review_flag = False

        # Handle Ditto / Arrows
        if cell_type in ["arrow", "copy"]:
            if last_valid_text:
                final_value = last_valid_text
            else:
                final_value = "COPY (NO PREVIOUS ROW)"
                review_flag = True

        # Handle Fixed Symbols
        elif cell_type in symbol_map:
            final_value = symbol_map[cell_type]

        # Handle Pure Text
        else:
            final_value = cloud_text
            if final_value.strip() != "":
                last_valid_text = final_value
            else:
                review_flag = True

        record = {
            "row": row_num,
            "easyocr_raw": item.get("easyocr_raw"),
            "cloud_vision_raw": item.get("cloud_vision_raw"),
            "detected_type": cell_type,
            "final_value": final_value,
            "requires_review": review_flag
        }
        resolved_records.append(record)
        print(f"Row {row_num:02d} | Final Value: '{final_value:<20}' | Type: {cell_type:<12} | Review: {review_flag}")

    # # Export to JSON
    # json_path = f"{output_name}.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(resolved_records, f, indent=4)
        
    # # Export to CSV
    # csv_path = f"{output_name}.csv"
    # if resolved_records:
    #     headers = resolved_records[0].keys()
    #     with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    #         writer = csv.DictWriter(f, fieldnames=headers)
    #         writer.writeheader()
    #         writer.writerows(resolved_records)
            
    # print(f"\n[Saved] Results exported successfully to:\n - {csv_path}\n - {json_path}")
    return resolved_records

def main():
    parser = argparse.ArgumentParser(description="Dual OCR + MobileNet Pipeline")
    parser.add_argument("-i", "--image", required=True, help="Path to the image.")
    parser.add_argument("-j", "--json", required=True, help="Path to COCO JSON")
    parser.add_argument("-c", "--column", required=True, help="Column to process")
    args = parser.parse_args()

    if not os.path.exists(args.image) or not os.path.exists(args.json):
        print("[Error] Image or JSON file not found.")
        return

    img = cv2.imread(args.image)
    with open(args.json, 'r') as f:
        data = json.load(f)
        
    extracted_data = dual_ocr_extraction(args.column, img, data, offset_y=0)
    resolve_and_save_pipeline_results(extracted_data, output_name=f"{args.column}_cloud_comparison")

if __name__ == "__main__":
    main()