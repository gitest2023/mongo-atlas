import numpy as np
from typing import List
import cv2


class Minutia:
    def __init__(self, x: int, y: int, angle: float, type: str):
        self.x = x
        self.y = y
        self.angle = angle
        self.type = type

class FingerPrint2:

    @classmethod
    def matching(cls, image_path1: str, image_path2: str):
        # Extract the minutiae from the two images
        minutiae1 = cls.__extract_minutiae(image_path1)
        minutiae2 = cls.__extract_minutiae(image_path2)

        return cls.__match(minutiae1, minutiae2)

    @classmethod
    def __distance(cls, m1: Minutia, m2: Minutia) -> float:
        # Calculate the Euclidean distance between two minutiae
        dx = m1.x - m2.x
        dy = m1.y - m2.y
        return np.sqrt(dx * dx + dy * dy)

    @classmethod
    def __angle_difference(cls, a1: float, a2: float) -> float:
        # Calculate the difference between two angles, in degrees
        return np.abs((a1 - a2 + 180) % 360 - 180)

    @classmethod
    def __match_score(cls, m1: Minutia, m2: Minutia) -> float:
        # Calculate the matching score between two minutiae
        d = cls.__distance(m1, m2)
        da = cls.__angle_difference(m1.angle, m2.angle)
        if d > 20 or da > 30:
            return 0.0
        else:
            return 1.0 - 0.1 * d - 0.4 * da / 30.0

    @classmethod
    def __match(cls, minutiae1: List[Minutia], minutiae2: List[Minutia]) -> float:
        # Calculate the matching score between two sets of minutiae
        score = 0.0
        for m1 in minutiae1:
            best_score = 0.0
            for m2 in minutiae2:
                s = cls.__match_score(m1, m2)
                if s > best_score:
                    best_score = s
            score += best_score
        return score / len(minutiae1)

    @classmethod
    def __extract_minutiae(cls, image_path: str) -> List[Minutia]:
        # Load the fingerprint image and convert it to grayscale
        image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to create a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find the contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the minutiae from the contours
        minutiae = []
        for cnt in contours:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, _, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    angle = np.arctan2(end[1]-start[1], end[0]-start[0]) * 180 / np.pi
                    minutia = Minutia(start[0], start[1], angle, "ending")
                    minutiae.append(minutia)
                    if d > 100:
                        far = tuple(cnt[int((s+e)/2)][0])
                        minutia = Minutia(far[0], far[1], angle, "bifurcation")
                        minutiae.append(minutia)

        return minutiae
