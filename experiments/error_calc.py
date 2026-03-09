
def translation_error_L2(t_gt, t_est):
    """Euclidean translation error in meters."""
    return float(np.linalg.norm(t_est - t_gt))


def rotation_error_geodesic(R_gt, R_est):
    """Geodesic rotation error in degrees."""
    R = R_gt.T @ R_est
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))

def rvec_tvec_to_R_t(rvec, tvec):
    """Convert OpenCV rvec/tvec to (R, t) where X_cam = R X_tag + t."""
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3).astype(np.float64)
    return R.astype(np.float64), t
