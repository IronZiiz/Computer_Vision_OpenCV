import cv2 
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

camera = cv2.VideoCapture(1)
resolution_x = 1280
resolution_y = 720

camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_y)

#

while True:
   success, frame = camera.read()
   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


   results = hands.process(frame_rgb)

   if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

   print(success)

   cv2.imshow("Img", frame)

   tecla = cv2.waitKey(1)
   if tecla == 27:
      break
