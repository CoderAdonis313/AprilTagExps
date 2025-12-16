import cv2
import numpy as np

########################### CONFIG ###################
TAG_SIZE = 0.3
CAM_MATRIX = np.array([
        [
            948.549417779676,
            0,
            570.3494303121358
        ],
        [
            0,
            940.9432002988655,
            343.967467586877
        ],
        [
            0,
            0,
            1
        ]
    ], dtype=np.float64)
DIST_COEFFS = np.array([
        0.029253537514305077,
        0.43901198364900257,
        -0.018038744110287074,
        -0.03409154715870114,
        -0.7497248615048877
    ], dtype=np.float64)

import numpy as np
import cv2

def pose_from_aruco_corners(corner_2d, tag_size, K, dist):
    """
    corner_2d: shape (4,2) float32 from detector (one marker)
    returns: rvec (3,1), tvec (3,1)
    """
    s = tag_size / 2.0

    # 3D object points of the tag corners in tag coordinate frame (centered)
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



########################### MAIN #####################

def main():
    # DETECTOR DICTIONARY
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

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('DETECTOR_WINDOW')

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
                print(corner)

                c2d = corner[0]  # (4,2)
                rvec, tvec = pose_from_aruco_corners(c2d, TAG_SIZE, CAM_MATRIX, DIST_COEFFS)
                if tvec is None:
                    continue

                x, y, z = tvec.flatten()
                print(f"id={id}  x={x:.3f}m  y={y:.3f}m  z={z:.3f}m")

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
    cv2.destroyWindow('DETECTOR_WINDOW')

    
if __name__ == '__main__':
    main()