import cv2

def main():
    # DETECTOR DICTIONARY
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    
    # DETECTOR PARAMS
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

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
            for corner, id in zip(corners, ids):
                print('_________________DETECT_______________')
                print(corner)

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