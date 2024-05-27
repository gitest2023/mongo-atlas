import cv2
import fingerprint_feature_extractor

# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
img = cv2.imread('image_path', 0)

FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
    img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True
)