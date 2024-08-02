import cv2
import mediapipe as mp
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# Initialize MediaPipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load stickers for specific emotions
stickers = {
    'happy': Image.open('happy.png'),
    'sad': Image.open('sad.png'),
    'neutral': Image.open('nautral.png')
}

# Function to detect face and landmarks
def detect_face_landmarks(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return [], []

        face_landmarks = []
        for detection in results.detections:
            face_landmarks.append(detection.location_data.relative_keypoints)

        return results.detections, face_landmarks

# Function to analyze emotion
def analyze_emotion(image):
    results = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)

    # Since results might be a list, we'll check if it's a list and get the first result
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    return result['dominant_emotion']

# Function to overlay sticker
def overlay_sticker(frame, sticker, position):
    sticker = sticker.resize((200, 200))  # Resize sticker as needed
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Ensure sticker has an alpha channel
    if sticker.mode != 'RGBA':
        sticker = sticker.convert('RGBA')

    # Create a mask for the sticker
    mask = sticker.split()[3]

    # Paste the sticker onto the frame using the mask
    frame_pil.paste(sticker, position, mask)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Main function to process the video feed
def process_video():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections, landmarks = detect_face_landmarks(frame)

        for detection in detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face_img = frame[y:y+h, x:x+w]
            dominant_emotion = analyze_emotion(face_img)

            # Determine sticker based on emotion
            sticker = stickers.get(dominant_emotion, None)

            if sticker:
                frame = overlay_sticker(frame, sticker, (x, y))

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
