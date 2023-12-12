from roboflow import Roboflow
import cv2 as cv


rf = Roboflow(api_key="eZdzAwikjCP3QUN6iDFF")
project = rf.workspace().project("tennis-balls-rai2k")
model = project.version(11).model

# infer on a local image
print(model.predict("tennisball.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("tennisball.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())





# videoCapture = cv.VideoCapture(0)


# while True:
#     ret, frame = videoCapture.read()
#     if not ret: break

#     result = model.predict(frame, confidence=60, overlap=30).json()
    
#     print(result)

#     if cv.waitKey(1) & 0xFF == ord('q'): break


# videoCapture.release()
# cv.destroyAllWindows()