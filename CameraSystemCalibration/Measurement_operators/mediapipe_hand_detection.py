# find help under:
# https://www.youtube.com/watch?v=01sAkU_NvOY
# or https://google.github.io/mediapipe/solutions/hands.html
import cv2

def findHands(mp_hands, mp_drawing_styles, mp_drawing, img, rgb, draw=True):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


if __name__ == '__main__':
    print("unit for hand detection \n \t -> run facial_landmarks_pose_detection.py  \n \t    with option: measure_hands=True")