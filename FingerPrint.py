import cv2
import numpy as np
import os

class FingerPrint:

    @classmethod
    def match(cls, img1_path, img2_path, thresh: float=0.95):
        """
        Matches two fingerprint images and checks if they are similar.

        Parameters:
        img1_path (str): The file path of the first fingerprint image.
        img2_path (str): The file path of the second fingerprint image.

        Returns:
        None
        """
        test_original = cv2.imread(img1_path) if isinstance(img1_path, str) else img1_path
        fingerprint_database_image = cv2.imread(img2_path) if isinstance(img2_path, str) else img2_path

        sift = cv2.xfeatures2d.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints_count = min(len(keypoints_1), len(keypoints_2))
        match_ratio = len(match_points) / keypoints_count

        if match_ratio > thresh:
            similarity = len(match_points) / keypoints_count * 100
            result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None)
            return {
                'similarity': similarity,
                'keypoint_image': result
            }

        return {
            'similarity': 0,
            'keypoint_image': None
        }

    @classmethod
    def search(cls, test_image_path, database_folder, thresh: float=0.95):
        """
        Matches a fingerprint image with a database of fingerprint images.

        Parameters:
        test_image_path (str): The file path of the fingerprint image to match.
        database_folder (str): The folder containing the database of fingerprint images.

        Returns:
        None
        """
        test_original = cv2.imread(test_image_path) if isinstance(test_image_path, str) else test_image_path

        for file in os.listdir(database_folder):
            fingerprint_database_image = cv2.imread(os.path.join(database_folder, file))

            sift = cv2.xfeatures2d.SIFT_create()

            keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

            matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)

            match_points = []
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints_count = min(len(keypoints_1), len(keypoints_2))
            match_ratio = len(match_points) / keypoints_count

            if match_ratio > thresh:
                similarity = len(match_points) / keypoints_count * 100
                result = cv2.drawMatches(test_original, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None)
                return {
                'found': True,
                'similarity': similarity,
                'keypoint_image': result,
                'found_image': file,
            }

        return {
                'found': False,
                'similarity': 0,
                'image': ''
            }