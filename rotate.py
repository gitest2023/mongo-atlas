
import cv2
import imutils
import numpy as np

import math
from typing import Tuple, Union
from deskew import determine_skew

def _rotate(image, angle: float=0, scale: float=1):
    """_summary_

    Args:
        image (MatLike): Input image
        angle (float, optional): Rotate angle. Defaults to 0. If angle > 0, rotate clockwise, otherwise, counterclockwise
        scale (float, optional): Zoom in rotated image. Defaults to 1.

    Returns:
        MatLike: Rotated Image
    """
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    return cv2.warpAffine(image, rotation_matrix, (height, width))
    return cv2.warpAffine(image, rotation_matrix)

def get_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate(image, tl: tuple=(), tr: tuple=(), br: tuple=(), bl: tuple=()):

    angle = get_angle(tl, tr)
    print('angle: ' + str(angle))
    return imutils.rotate_bound(image, angle)

def rotate2(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

image = cv2.imread('images/idcard/idcard-60.png')
# rotated_image = rotate(image, tl=[10, 20], tr=[5,25])
# rotated_image = imutils.rotate_bound(image, -60)
# cv2.imshow("Rotated", rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
print(angle)
# rotated = rotate(image, angle, (0, 0, 0))
rotated = rotate2(image, angle, (0, 0, 0))
cv2.imwrite('output.png', rotated)