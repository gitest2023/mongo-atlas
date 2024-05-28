
import cv2
import imutils
import numpy as np

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

image = cv2.imread('images/test-6-052098011479-l.jpg')
rotated_image = rotate(image, tl=[10, 20], tr=[5,25])
cv2.imshow("Rotated", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
