import cv2 
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

camera = cv2.VideoCapture(1)
resolution_x = 1280
resolution_y = 720

camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

#Canvas 
canvas = None
prev_x, prev_y = None, None

# style 
points = mp_drawing.DrawingSpec(color=(255, 200, 100), thickness=3, circle_radius=4)
conexions = mp_drawing.DrawingSpec(color=(255, 200, 100), thickness=2, circle_radius=2)

while True:
   success, frame = camera.read()
   frame = cv2.flip(frame, 1)
   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   if canvas is None:
        canvas = np.zeros_like(frame)

   results = hands.process(frame_rgb)

   if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=points,
                                    connection_drawing_spec=conexions)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 20)

            prev_x, prev_y = x, y
   else:
                    prev_x, prev_y = None, None 
                    canvas[:] = 0

   canvas = cv2.addWeighted(canvas, 0.99, np.zeros_like(canvas), 0.01, 0)
   frame_out = cv2.addWeighted(frame, 1, canvas, 1, 0)

   cv2.imshow("Img", frame_out)

   tecla = cv2.waitKey(1)
   if tecla == 27:
      break
