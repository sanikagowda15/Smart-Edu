# SMART EDU - COMPUTER VISION / OCR

import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def detect_topic(text):
    text_lower = (text or "").lower()

    topics = {
        "Python": [
            "python", "string", "list", "tuple", "dictionary",
            "function", "loop", "variable", "print(", "def ", "import "
        ],
        "Machine Learning": [
            "machine learning", "classification", "regression",
            "training", "dataset", "prediction", "supervised learning"
        ],
        "Data Analytics": [
            "data analytics", "pandas", "data analysis",
            "visualization", "matplotlib"
        ],
        "Computer Vision": [
            "computer vision", "image processing",
            "opencv", "object detection"
        ],
        "Artificial Intelligence": [
            "artificial intelligence", "intelligent system",
            "knowledge representation"
        ],
        "Mathematics": [
            "calculate", "solve", "equation", "integral",
            "differentiate", "probability", "matrix", "theorem",
            "find the value", "evaluate"
        ],
        "Database": [
            "sql", "database", "query", "normalization",
            "primary key", "foreign key"
        ],
        "Web Development": [
            "html", "css", "javascript", "http", "web page", "dom"
        ],
        "Computer Networks": [
            "computer network", "tcp", "udp", "ip address",
            "routing", "osi model", "protocol"
        ],
        "Operating Systems": [
            "operating system", "process scheduling", "deadlock",
            "memory management", "paging", "semaphore"
        ]
    }

    scores = {
        topic: sum(1 for keyword in keywords if keyword in text_lower)
        for topic, keywords in topics.items()
    }

    best_topic = max(scores, key=scores.get)

    return best_topic if scores[best_topic] > 0 else "General"


def _ocr_candidate(image, config):
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=Output.DICT
    )

    lines = {}
    confidences = []

    for i, raw_text in enumerate(data["text"]):
        word = raw_text.strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not word:
            continue

        if confidence >= 0:
            confidences.append(confidence)

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        left = int(data["left"][i])
        lines.setdefault(key, []).append((left, word))

    output_lines = []

    for words in lines.values():
        words.sort(key=lambda item: item[0])
        output_lines.append(" ".join(word for _, word in words))

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    return "\n".join(output_lines).strip(), average_confidence


def extract_text_in_order(image):
    if image is None:
        return ""

    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    candidates = []

    try:
        candidates.append(
            _ocr_candidate(enlarged, "--oem 3 --psm 6")
        )
    except pytesseract.TesseractError:
        pass

    try:
        adaptive = cv2.adaptiveThreshold(
            enlarged,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        candidates.append(
            _ocr_candidate(adaptive, "--oem 3 --psm 6")
        )
    except pytesseract.TesseractError:
        pass

    try:
        candidates.append(
            _ocr_candidate(enlarged, "--oem 3 --psm 3")
        )
    except pytesseract.TesseractError:
        pass

    if not candidates:
        return ""

    best_text, _ = max(candidates, key=lambda item: item[1])
    return best_text


def analyze_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image. "
            "Please upload a valid image file."
        )

    height, width, channels = image.shape

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = extract_text_in_order(image)
    detected_topic = detect_topic(extracted_text)

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "average_brightness": round(float(gray_image.mean()), 2),
        "extracted_text": extracted_text,
        "detected_topic": detected_topic
    }