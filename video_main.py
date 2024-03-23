from functions import *


def detect_treasure(frame):
    frame = cv2.resize(frame, (640*2, 480*2))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    ret, frame_bin = cv2.threshold(frame_blur, 80, 255, cv2.THRESH_BINARY)
    locating_points = find_locating_point(frame_bin)
    if locating_points is not None:
        for locating_point in locating_points:
            cv2.circle(frame, locating_point, 2, (0, 255, 0), 3)
        frame_trans = get_trans(frame, locating_points)
        frame_roi = get_roi(frame_trans)
        frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2GRAY)
        ret, frame_roi_bin = cv2.threshold(frame_roi_gray, 80, 255, cv2.THRESH_BINARY)
        circles = find_circles(frame_roi_bin)
        if circles is not None:
            for circle in circles:
                x = int((circle[0] - 99) / (510 / 8) + 1)
                y = int((circle[1] - 99) / (510 / 8) + 1)
                if 0 < x < 9 and 0 < y < 9:
                    cv2.circle(frame_roi, circle, 2, (0, 0, 255), 3)
                    cv2.putText(frame_roi, str((x, y)), (circle[0] - 10, circle[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 255), thickness=1)
            cv2.imshow("video", frame_roi)
            cv2.waitKey(0)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        flag, frames = cap.read()
        detect_treasure(frames)
        if not flag:
            break
        cv2.imshow("video", frames)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
