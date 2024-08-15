# find help under:
# https://www.youtube.com/watch?v=01sAkU_NvOY
# or https://google.github.io/mediapipe/solutions/pose.html

def findPose(mp_pose, mp_drawing_styles, mp_drawing, img, rgb, draw=True):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


if __name__ == '__main__':
    print("unit for human pose detection \n \t -> run facial_landmarks_pose_detection.py  \n \t    with option: measure_pose=True")
