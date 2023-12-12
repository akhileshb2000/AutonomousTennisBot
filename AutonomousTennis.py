from  pycreate2 import Create2
import time
from roboflow import Roboflow
import cv2 as cv

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
TENNIS_WIDTH = 2.5 #INCHES


CAM_X = 1920
CAM_Y = 1080
CENTER_THRESHOLD = 450
CAM_CENTER_X = CAM_X / 2.0
CAM_CENTER_Y = CAM_Y / 2.0
BASE_SPEED = 25

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


def main():
    model = initVision()
    bot = initBot()

    #Start the webcam to run the vision
    videoCapture = cv.VideoCapture(0)

    start_time = time.time()

    while True:
        ret, frame = videoCapture.read()
        if not ret: break

        result = model.predict(frame, confidence=60, overlap=30).json()
        
        tennisball_centers = getBoundingBoxCenters(result)

        if tennisball_centers:
            start_time = time.time()
            ball = tennisball_centers[0]
            
            focal_ball = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, result['predictions'][0]['width'])
            distance = distance_finder(focal_ball, TENNIS_WIDTH, result['predictions'][0]['width'])
            
            if ball[0] < CAM_CENTER_X - CENTER_THRESHOLD:
                dist = 1.5 * ((ball[0] / (CAM_CENTER_X - CENTER_THRESHOLD)) ** -1)
                bot.drive_direct(int(dist * BASE_SPEED), BASE_SPEED)
                print("Curving left: ", BASE_SPEED, ", ", dist * BASE_SPEED)
            elif ball[0] > CAM_CENTER_X + CENTER_THRESHOLD:
                dist = 1.5 * (ball[0] / (CAM_CENTER_X - CENTER_THRESHOLD))
                bot.drive_direct(BASE_SPEED, int(dist * BASE_SPEED))
                print("Curving right: ", dist * BASE_SPEED, ", ", BASE_SPEED)
            else:
                bot.drive_direct(8 * BASE_SPEED, 8 * BASE_SPEED)
                print("Straight Ahead")
            print("Distance: ", distance, " inches")

        else:
            if time.time() - start_time > 3:
                bot.drive_direct(2 * BASE_SPEED, 2 * -BASE_SPEED)
                print("scanning...")
            else:
                bot.drive_stop()
                
 

        #cv.imshow("AutonomousTennis", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break


    videoCapture.release()
    cv.destroyAllWindows()

def getBoundingBoxCenters(model_result):
    predictions = model_result['predictions']

    centers = []

    for pred in predictions:
        center = getCenter(pred['x'], pred['y'], pred['width'], pred['height'])
        centers.append(center)

    return centers

def getCenter(x, y, width, height):
    center = (x + width/2.0, y - height/2.0)
    return center

def initVision():
    # Import the roboflow YOLOv8 trained tennis ball CV model
    rf = Roboflow(api_key="-- Insert API Key --")
    project = rf.workspace().project("tennis-balls-rai2k")
    model = project.version(11).model
    return model

def initBot():
     # Create a Create2.
    port = "/dev/tty.usbserial-DN026AF1" 
    bot = Create2(port)

    # Start the Create 2
    bot.start()
    # Put the Create2 into 'safe' mode so we can drive it
    bot.safe()

    return bot


main()
