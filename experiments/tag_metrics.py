import cv2
import numpy as np
import json
from argparse import ArgumentParser


def read_calib(c_path):
    with open(c_path, 'r') as f:
        calib = json.load(f)

    cam_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
    dist = np.array(calib['distortion_coefficients'], dtype=np.float64)
    return cam_matrix, dist


def read_gt(gt_path):
    """
    Expected JSON:
      { "R": [[...],[...],[...]], "t": [x,y,z], "direction": "tag2cam" }
    direction can be 'tag2cam' or 'cam2tag'
    """
    with open(gt_path, "r") as f:
        gt = json.load(f)

    R = np.array(gt["R"], dtype=np.float64)
    t = np.array(gt["t"], dtype=np.float64).reshape(3)

    direction = gt.get("direction", "tag2cam")
    if direction == "cam2tag":
        # invert cam->tag to tag->cam
        R = R.T
        t = -R @ t

    return R, t


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


def cam_loop(detector, calib, TAG_SIZE):
    cam = cv2.VideoCapture(0)
    # cv2.namedWindow('DETECTOR_WINDOW')
    CAM_MATRIX, DIST_COEFFS = calib

    while cam.isOpened():
        ok, frame = cam.read()

        if not ok:
            print('Camera not working')
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray_img)
        res_img = frame.copy()

        if ids is not None:
            # rvecs, tvecs, _objPoints = detector.
            # estimatePoseSingleMarkers(
            #     corners, TAG_SIZE, CAM_MATRIX, DIST_COEFFS
            # ) 

            for i, (corner, id) in enumerate(zip(corners, ids)):
                print('_________________DETECT_______________')

                c2d = corner[0]  # (4,2)
                rvec, tvec = pose_from_aruco_corners(c2d, TAG_SIZE, CAM_MATRIX, DIST_COEFFS)
                if tvec is None:
                    continue

                x, y, z = tvec.flatten()
                print(f"id={id}  x={x:.3f}m  y={y:.3f}m  z={z:.3f}m")
                print(rvec)

                # Draw axes on the tag (helps sanity-check pose)
                cv2.drawFrameAxes(res_img, CAM_MATRIX, DIST_COEFFS, rvec, tvec, TAG_SIZE * 0.5)

                pt1, pt2, pt3, pt4 = [tuple(map(int, pt)) for pt in corner[0]]
                id = id[0]

                cv2.line(res_img, pt1, pt2, (0, 255, 0), 5)
                cv2.line(res_img, pt2, pt3, (0, 255, 0), 5)
                cv2.line(res_img, pt3, pt4, (0, 255, 0), 5)
                cv2.line(res_img, pt4, pt1, (0, 255, 0), 5)

                fontp = ((pt1[0] - 10), (pt1[1] - 10))
                midp = ((pt1[0] + pt3[0]) // 2, (pt1[1] + pt3[1]) // 2)
                cv2.putText(res_img, f'id:{id}', fontp, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
                cv2.circle(res_img, midp, 2, (0, 0, 255), 5)

        cv2.imshow('DETECTOR_WINDOW', res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    # cv2.destroyWindow('DETECTOR_WINDOW')


def main():
    parser = ArgumentParser()
    parser.add_argument('calib_file', type=str)
    parser.add_argument('gt_file', type=str)
    args = parser.parse_args()

    TAG_SIZE = 0.3
    calib = read_calib(args.calib_file)
    gt = read_gt(args.gt_file)
    detector = setup_detector()
    cam_loop(detector, calib, TAG_SIZE)


if __name__ == '__main__':
    main()