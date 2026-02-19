from core.geometry import calculate_angle_3d
class BodyState:
    """
    The Biomechanics Abstraction Layer.

    Wraps raw mediapipe points, and consoldates them into bodyparts both the developer and computer can understand.
    """
    def __init__(self, mp_landmarks):
        self.landmarks = mp_landmarks.landmark
    
    #--All hardcoded body parts are in this file.
    #--If pose estimation model changes, change values in this file.

    @property
    def left_shoulder(self):
        return self.landmarks[11]
    
    @property
    def right_shoulder(self):
        return self.landmarks[12]
    
    @property
    def left_elbow(self):
        return self.landmarks[13]
    
    @property
    def right_elbow(self):
        return self.landmarks[14]
    
    @property
    def left_wrist(self):
        return self.landmarks[15]
    
    @property
    def right_wrist(self):
        return self.landmarks[16]
    
    @property
    def left_knee(self):
        return self.landmarks[25]

    @property
    def left_hip(self):
        return self.landmarks[23]

    @property
    def right_knee(self): 
        return self.landmarks[23]
    
    @property
    def right_hip(self):
        return self.landmarks[24]
    
    @property
    def left_ankle(self):
        return self.landmarks[27]

    @property
    def right_ankle(self):
        return self.landmarks[28]
    
    # --- Angle Calculaions --- #

    """
    Calculated angles of different regions of the body
    180 degrees = fully straight
    90 degrees = fully bent
    """

    @property
    def left_elbow_angle(self):
        return calculate_angle_3d(
            self.left_shoulder,
            self.left_elbow,
            self.left_wrist,
        )
    
    @property
    def right_elbow_angle(self):
        return calculate_angle_3d(
            self.right_shoulder,
            self.right_elbow,
            self.right_wrist,
        )
    
    @property
    def left_hip_angle(self):
        return calculate_angle_3d(
            self.left_shoulder,
            self.left_hip,
            self.left_knee,
        )
    
    @property
    def right_hip_angle(self):
        return calculate_angle_3d(
            self.right_shoulder,
            self.right_hip,
            self.right_knee,
        )

    @property
    def left_shoulder_angle(self):
        return calculate_angle_3d(
            self.left_hip,
            self.left_shoulder,
            self.left_elbow,
        )

    @property
    def right_shoulder_angle(self):
        return calculate_angle_3d(
            self.right_hip,
            self.right_shoulder,
            self.right_elbow,
        )

    @property
    def left_knee_angle(self):
        return calculate_angle_3d(
            self.left_hip,
            self.left_knee,
            self.left_ankle,
        )

    @property
    def right_knee_angle(self):
        return calculate_angle_3d(
            self.right_hip,
            self.right_knee,
            self.right_ankle,
        )


