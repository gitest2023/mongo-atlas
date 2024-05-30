import cv2
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()

# images = [
#     keras_ocr.tools.read(url) for url in [
#         'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
#         'https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg',
#         'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
#     ]
# ]
images = [
    cv2.imread('images/ocr/gks-1.jpg')
]
# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)
print(prediction_groups)
# Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)