from vision_bridge import VisionBridge
import cv2

# Webcam
# with VisionBridge(
#     source=0,                                  # webcam index
#     pose_model_path=r"pose_landmarker.task",
#     hand_model_path=r"hand_landmarker.task",
# ) as bridge:
#     for frame, actions in bridge.run():
#         for action in actions:
#             print(action.as_tuple())           # ("gesture", "fan", "toggle")
#         cv2.imshow("Bridge", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break



# # ESP32-CAM stream

with VisionBridge(
    source="http://10.134.10.29/stream",    # your ESP32-CAM IP
    pose_model_path="pose_landmarker.task",
    hand_model_path="hand_landmarker.task",
) as bridge:
    for frame, actions in bridge.run():
        for action in actions:
            print(action.as_tuple())           # ("gesture", "fan", "toggle")
        cv2.imshow("Bridge", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break