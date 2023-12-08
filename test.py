import cv2

import numpy as np

image = cv2.imread('images/shirt.jpeg')
image = cv2.resize(image, (640, 480))

x1, y1 = 320, 0
x2, y2 = 320, 480


def distance_from_line(x, y):
    numerator = (x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if denominator == 0:
        return 0
    return numerator / denominator


cv2.line(image, (x1, y1), (x2, y2) , (0, 255, 0), 2)

cv2.circle(image, (160, 240), 5, (0, 0, 255), -1)

dis = distance_from_line(160, 240)
if dis > 0:
    side = "R"
elif dis < 0:
    side = "L"
else:
    side = "O"

cv2.putText(image, side, (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.circle(image, (480, 240), 5, (0, 0, 255), -1)

dis = distance_from_line(480, 240)
if dis > 0:
    side = "R"
elif dis < 0:
    side = "L"
else:
    side = "O"

cv2.putText(image, side, (480, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
