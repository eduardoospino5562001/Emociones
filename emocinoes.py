import fer
import cv2
import time

class EmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detector = fer.FER(mtcnn=True)

    def detect_emotions(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect emotions in the frame
            emotions = self.face_detector.detect_emotions(frame)

            # If emotions are detected
            if emotions:
                # Get the emotion with the highest probability
                emotion, score = max(emotions[0]['emotions'].items(), key=lambda x: x[1])

                # Display the emotion on the screen
                cv2.putText(frame, f"Emotion: {emotion}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Emotion Detection', frame)

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Limit the frame rate to 30 FPS
            time.sleep(1 / 30)

        # Release the camera and close the window
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.detect_emotions()