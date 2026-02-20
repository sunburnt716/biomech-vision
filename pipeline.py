import cv2
import mediapipe as mp
from domain.body_state import BodyState
from core.geometry import is_valid_human_pose
from core.signal import WristValleyDetector



def main():
    #--Setup
    video_path = "apps/data/Good Examples/Virat Kohli Side View.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    #Initialize Mediapipe Pose with the stricter confidenence thresholds
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7,
    )

    #Developed time-series heuristic
    valley_detector = WristValleyDetector(history_size = 5)

    print("Pipeline Running... Waiting for impact. Press 'q' to Quit")

    #--Ingestion Loop--
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            #Abstraction.

            body = BodyState(results.pose_landmarks)

            if not is_valid_human_pose(body):
                continue
            
            #NOTE: Currently we are tracking the right wrist,
            #if batsmen is left handed, switch to left-handed
            #Eventually, this will be automated
            is_impact = valley_detector.process_frame(body.right_wrist.y)
            
            if is_impact:
                print(f"\n Impact Detected at Frame {current_frame}")

                #Extract biomechanics at moment of impact
                l_elbow = body.left_elbow_angle
                l_knee = body.left_knee_angle
                print(f"Metrics -> Lead Elbow: {l_elbow:.1f} degrees, Lead Kneww: {l_knee:.1f} degrees")

                cv2.putText(frame, "Impact Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                h, w, _ = frame.shape
                wrist_px = (int(body.right_wrist.x * w), int(body.right_wrist.y * h))
                cv2.circle(frame, wrist_px, 5, (0, 0, 255), -1)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)



            cv2.rectangle(frame, (10, 10), (350, 150), (0, 0, 0), -1)

            cv2.putText(frame, f"L Elbow: {body.left_elbow_angle: .1f} deg", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"R Elbow: {body.right_elbow_angle:.1f} deg", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"L Knee:  {body.left_knee_angle:.1f} deg", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)


            

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        cv2.imshow('Biomech-Vision-Pipeline', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


