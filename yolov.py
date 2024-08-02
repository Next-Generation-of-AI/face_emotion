import cv2
import dlib

# Load the trained model
model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
model.eval()

# Load the dlib face detector
detector = dlib.get_frontal_face_detector()

# Define a function to detect emotion
def detect_emotion(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Define a dictionary to map label indices to emotions
emotion_dict = {0: 'happy', 1: 'sad', 2: 'neutral'}

# Open a video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = gray[y:y+h, x:x+w]
        emotion_label = detect_emotion(face_image)
        emotion_text = emotion_dict[emotion_label]

        # Draw a rectangle around the face and write the emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
