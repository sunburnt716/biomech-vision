import numpy as np

def create_vector(origin, destination):
    """
    Converts two Mediapipe landmarks into a 3D numpy vector.
    Formula = Vector = Destination - Origin
    """

    return np.array([
        destination.x - origin.x,
        destination.y - origin.y,
        destination.z - origin.z,
    ])
def calculate_angle_3d(p1, p2, p3):
    """
    Calculates the 3D angle between three points.
    p2 is the vertex
    p1 and p3 are the endpoints

    Returns:
    float: The angle in degrees.
    """

    vector_a = create_vector(origin = p2, destination = p1)

    vector_b = create_vector(origin = p2, destination = p3)

    norm_a  = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    dot_product = np.dot(vector_a, vector_b)

    cos_theta = dot_product / (norm_a * norm_b)

    cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta_clipped)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)

def is_valid_human_pose(body, min_conf=0.4, min_torso_height=0.02, max_width_ratio=1.5):
    """
    Validates if the detected object is a human.
    Updated with leniency for side-profile camera angles and debugging.
    """
    l_sh = body.left_shoulder
    r_sh = body.right_shoulder
    l_hip = body.left_hip
    r_hip = body.right_hip

    # --- 1. The Side-Profile Friendly Confidence Check ---
    # Instead of requiring ALL 4 joints, we just need the camera 
    # to clearly see AT LEAST ONE shoulder and AT LEAST ONE hip.
    shoulder_visible = (l_sh.visibility > min_conf) or (r_sh.visibility > min_conf)
    hip_visible = (l_hip.visibility > min_conf) or (r_hip.visibility > min_conf)

    if not (shoulder_visible and hip_visible):
        print(f"DEBUG: Rejected due to low visibility. L_Sh:{l_sh.visibility:.2f}, R_Sh:{r_sh.visibility:.2f}")
        return False

    # --- 2. Geometry Calculation ---
    mid_shoulder_y = (l_sh.y + r_sh.y) / 2.0
    mid_hip_y = (l_hip.y + r_hip.y) / 2.0
    torso_height = abs(mid_hip_y - mid_shoulder_y)
    torso_width = abs(l_sh.x - r_sh.x)

    # --- 3. The "Dwarf" Filter ---
    if torso_height < min_torso_height:
        print(f"DEBUG: Rejected! Torso too small ({torso_height:.2f} < {min_torso_height})")
        return False

    # --- 4. The "Squish" Filter ---
    if torso_height == 0: 
        return False 
        
    current_ratio = torso_width / torso_height
    if current_ratio > max_width_ratio:
        print(f"DEBUG: Rejected! Torso too wide. Ratio: {current_ratio:.2f}")
        return False

    return True