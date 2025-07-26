import cv2
import numpy as np
import pytesseract
import re
import csv
import difflib
from ultralytics import YOLO
import os

# --------------------------------
# CONFIG
# --------------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

VIDEO_PATH = r'C:\Users\kusha\Desktop\automatic license plate recognisation\p2.mp4'
OUTPUT_VIDEO = r'C:\Users\kusha\Desktop\automatic license plate recognisation\predict_out.mp4'
MODEL_PATH = r'C:\Users\kusha\Desktop\automatic license plate recognisation\runs\detect\train\weights\last.pt'
CSV_PATH = r'C:\Users\kusha\Desktop\automatic license plate recognisation\detected_plates.csv'
SR_MODEL_PATH = 'LapSRN_x4.pb'

CONF_THRESH = 0.5

VALID_STATE_CODES = [
    'KA', 'MH', 'DL', 'TN', 'AP', 'TS', 'GJ', 'RJ', 'UP', 'MP', 'PB', 'HR', 'CH',
    'JK', 'UK', 'HP', 'WB', 'OD', 'BR', 'CG', 'KL', 'GA', 'TR', 'AS', 'AR', 'MN',
    'NL', 'SK', 'ML', 'MZ'
]

seen_plates = set()

# --------------------------------
# FUNCTIONS
# --------------------------------
def correct_state_code(raw):
    guess = raw[:2]
    best = difflib.get_close_matches(guess, VALID_STATE_CODES, n=1, cutoff=0.6)
    return best[0] if best else guess

def is_similar(p1, p2):
    return difflib.SequenceMatcher(None, p1, p2).ratio() > 0.85

def preprocess_plate(img):
    # CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def clean_ocr(raw):
    raw = raw.upper()
    raw = re.sub(r'[^A-Z0-9]', '', raw)
    for a, b in [('O','0'),('Q','0'),('I','1'),('|','1'),('L','1'),
                  ('B','8'),('S','5'),('Z','2'),('G','6')]:
        raw = raw.replace(a, b)
    return raw

def parse_plate(raw):
    # Remove all spaces/newlines just in case
    raw = raw.replace(' ', '').replace('\n', '')
    
    # Extract letters and digits
    letters = ''.join(re.findall(r'[A-Z]', raw))
    digits = ''.join(re.findall(r'\d', raw))

    if len(letters) < 2 or len(digits) < 1:
        return "🔴INVALID"

    # State code correction
    state = correct_state_code(letters[:2])

    # Try to find the district (1–2 digits)
    district_match = re.match(r'(\d{1,2})', digits)
    if not district_match:
        return "🔴INVALID"
    district = district_match.group(1)
    digits = digits[len(district):]

    # Series = remaining letters after state
    series = letters[2:]

    # Number = remaining digits
    number = digits

    if not number:
        return "🔴INVALID"

    return f"{state} {district} {series} {number}".strip()

# --------------------------------
# LOAD YOLO
# --------------------------------
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# --------------------------------
# LOAD Super-Resolution
# --------------------------------
print("[INFO] Loading Super-Resolution model...")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(SR_MODEL_PATH)
sr.setModel("lapsrn", 4)

# --------------------------------
# VIDEO SETUP
# --------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("❌ Could not open video.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

try:
    csvfile = open(CSV_PATH, 'w', newline='', encoding='utf-8')
except PermissionError:
    print(f"❌ Close the CSV file before running: {CSV_PATH}")
    exit()
csv_writer = csv.writer(csvfile)
csv_writer.writerow(['Frame', 'Plate'])

# --------------------------------
# MAIN LOOP
# --------------------------------
frame_num = 0
print("[INFO] Starting processing...")

while ret:
    results = model(frame)[0]
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = box
        if score < CONF_THRESH:
            continue

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Super-resolution
        try:
            crop = sr.upsample(crop)
        except:
            pass

        # Preprocessing
        processed = preprocess_plate(crop)

        # OCR with multiline flattening
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw = pytesseract.image_to_string(processed, config=config)
        raw = raw.replace('\n', '').replace(' ', '').upper()
        raw = clean_ocr(raw)
        plate = parse_plate(raw)

        # Deduplication
        duplicate = any(is_similar(plate, existing) for existing in seen_plates)
        if not duplicate and "INVALID" not in plate:
            seen_plates.add(plate)
            csv_writer.writerow([frame_num, plate])
            print(f"[✔] Frame {frame_num}: {plate}")

        # Annotate
        color = (0, 255, 0) if "INVALID" not in plate else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Debug
        cv2.imshow('Processed Plate', processed)

    out.write(frame)
    cv2.imshow('ANPR - Press Q to Quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    frame_num += 1

cap.release()
out.release()
csvfile.close()
cv2.destroyAllWindows()
print("✅ All done! Video & CSV saved.")
