from scipy.spatial import distance as dist

def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR) for a given eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for determining if eyes are open (focused)
EAR_THRESHOLD = 0.25
