import cv2

# DETECTOR DICTIONARY
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# DETECTOR PARAMETERS   
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod

# DETECTOR
detector = cv2.aruco.ArucoDetector(dictionary, params)

cap = cv2.VideoCapture(0)
cv2.namedWindow('DETECTOR_WINDOW')

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corner, ids, _ = detector.detectMarkers(gray_img)

    if ids is not None:
        print('_______________________DETECT_______________________')
        print(corner, ids)
        cv2.aruco.drawDetectedMarkers(frame, corner, ids)

    cv2.imshow('DETECTOR_WINDOW', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyWindow('DETECTOR_WINDOW')