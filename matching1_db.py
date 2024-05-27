import cv2

from FingerPrint import FingerPrint

img1 = cv2.imread("images/test-6-052098011479-l.jpg")
print(FingerPrint.search(img1, "fingerprints"))