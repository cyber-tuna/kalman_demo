import cv2 as cv
import numpy as np

cursor_x = 250
cursor_y = 250
start = False

img_height = 500
img_width = 500

img = np.zeros((img_height, img_width, 3), np.uint8)

kalman = cv.KalmanFilter(4, 2, 0, cv.CV_32F)

def update_cursor_pos(event, x, y, flags, param):
	global cursor_x, cursor_y, start, img
	if event == cv.EVENT_LBUTTONDOWN:
		start = True
		cursor_x = x
		cursor_y = y
		kalman.statePost = 1. * np.array([[cursor_x],[cursor_y],[0.],[0.]])
	elif event == cv.EVENT_LBUTTONDBLCLK:
		img = np.zeros((img_height, img_width, 3), np.uint8)
	elif event == cv.EVENT_MOUSEMOVE:
		cursor_x = x
		cursor_y = y


if __name__ == "__main__":

	cv.namedWindow("KalmanDemo")
	cv.setMouseCallback('KalmanDemo', update_cursor_pos)

	kalman.transitionMatrix = np.array([[1.,0.,.7,0.],[0.,1.,0.,.7],[0.,0.,1.,0.],[0.,0.,0.,1.]])
	kalman.measurementMatrix = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
	kalman.processNoiseCov = 1e-4 * np.eye(4)
	kalman.measurementNoiseCov = 15 * np.eye(2)
	kalman.errorCovPost = 1. * np.eye(4)

	while(True):
		if(start):
			measurement = 1. * np.array([[0.],[0.]])

			measured_x = cursor_x + np.random.randn(1, 1) * kalman.measurementNoiseCov[0,0]
			measured_y = cursor_y + np.random.randn(1, 1) * kalman.measurementNoiseCov[1,1]

			measurement[0] = 1. * measured_x 
			measurement[1] = 1. * measured_y

			prediction = kalman.predict()

			kalman.correct(measurement)
		
			cv.circle(img, (int(measurement[0]) , int(measurement[1])),2, (255, 255, 0), thickness=-1) 
			cv.circle(img, (int(kalman.statePost[0]) , int(kalman.statePost[1])),2, (0, 255, 255), thickness=-1) 

		cv.imshow("KalmanDemo", img)

		code = cv.waitKey(10)
		if code != -1:
			break

	cv.destroyWindow("Kalman")