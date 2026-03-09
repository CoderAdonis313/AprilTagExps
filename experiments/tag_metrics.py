import cv2
import numpy as np
import json
from argparse import ArgumentParser
from time import strftime, time


def read_calib(c_path):
    with open(c_path, 'r') as f:
        calib = json.load(f)

    cam_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
    dist = np.array(calib['distortion_coefficients'], dtype=np.float64)
    return cam_matrix, dist


def pose_from_aruco_corners(corner_2d, tag_size, K, dist):
    """
    corner_2d: shape (4,2) float32 from detector (one marker)
    returns: rvec (3,1), tvec (3,1)
    """
    s = tag_size / 2.0

    obj_pts = np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0],
    ], dtype=np.float32)

    img_pts = corner_2d.astype(np.float32)

    # Best for planar squares if available, else fall back
    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flag)
    if not ok:
        return None, None
    return rvec, tvec


def setup_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    
    # DETECTOR PARAMS
    params = cv2.aruco.DetectorParameters()
    # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    ####################### PARAMS ########################

    # Thresholding robustness
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7

    # Allow smaller markers (if your tag is small in frame)
    params.minMarkerPerimeterRate = 0.01  # default is often higher

    # A bit more tolerant quad approximation
    params.polygonalApproxAccuracyRate = 0.04

    # Corner refinement (jitter improvement, not detection rate)
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    # DETECTOR
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector


def euler_zyx_deg_from_R(R):
    """
    Returns roll, pitch, yaw in DEGREES using ZYX convention:
      R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    # yaw (Z)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    # pitch (Y)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # roll (X)
    roll = np.arctan2(R[2, 1], R[2, 2])

    return (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))


def view_angle_deg(R, tvec):
    """
    Angle between tag normal and camera viewing direction (acute).
    Tag normal is +Z in tag frame; in camera frame it's R[:,2].
    Viewing direction (tag->camera) is -t/||t||.
    """
    n_cam = R[:, 2]
    t = tvec.reshape(3)
    d = np.linalg.norm(t)
    if d < 1e-9:
        return 0.0
    v_cam = -t / d
    phi = np.degrees(np.arccos(np.clip(np.dot(n_cam, v_cam), -1.0, 1.0)))
    return float(min(phi, 180.0 - phi))


def cam_loop(detector, K, dist, out_txt, tag_size, cam_index=0, expected_id=None):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")


    with open(out_txt, "w") as f:
        f.write("timestamp x y z roll pitch yaw view_angle_deg\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None or len(ids) == 0:
                continue

            # choose tag: expected_id if provided else first one
            chosen_idx = 0
            if expected_id is not None:
                ids_flat = ids.reshape(-1)
                matches = np.where(ids_flat == expected_id)[0]
                if len(matches) == 0:
                    continue
                chosen_idx = int(matches[0])

            c2d = corners[chosen_idx][0]
            rvec, tvec = pose_from_aruco_corners(c2d, tag_size, K, dist)
            if tvec is None:
                continue

            R, _ = cv2.Rodrigues(rvec)
            x, y, z = map(float, tvec.reshape(3))
            roll, pitch, yaw = euler_zyx_deg_from_R(R)
            phi = view_angle_deg(R, tvec)
            ts = time()

            f.write(f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f} "
                    f"{roll:.3f} {pitch:.3f} {yaw:.3f} {phi:.3f}\n")
            f.flush()

    cap.release()


def main():
    parser = ArgumentParser()
    parser.add_argument('calib_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument("--tag_size", type=float, default=0.30)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--expected_id", type=int, default=None)
    args = parser.parse_args()

    detector = setup_detector()
    K, dist = read_calib(args.calib_file)
    detector = setup_detector()
    out_txt = f"{args.out_file}/out_{strftime('%H_%M_%S')}.txt"
    cam_loop(detector, K, dist, out_txt, args.tag_size, cam_index=args.cam, expected_id=args.expected_id)


if __name__ == '__main__':
    main()