import sys
from Xlib import display
PY3 = sys.version_info[0] == 3

if PY3:
    long = int

import cv2 as cv
from math import cos, sin, sqrt
import numpy as np

cursor_x = 250
cursor_y = 250
start = False

img_height = 500
img_width = 500

img = np.zeros((img_height, img_width, 3), np.uint8)

def update_cursor_pos(event, x, y, flags, param):
	global cursor_x, cursor_y, start, img
	if event == cv.EVENT_LBUTTONDOWN:
		start = True
	elif event == cv.EVENT_LBUTTONDBLCLK:
		img = np.zeros((img_height, img_width, 3), np.uint8)
	elif event == cv.EVENT_MOUSEMOVE:
		cursor_x = x
		cursor_y = y


if __name__ == "__main__":

	# cursor_x, cursor_y, img
	kalman = cv.KalmanFilter(4, 2, 0, cv.CV_32F)

	code = long(-1)

	# code = long(-1)

	cv.namedWindow("KalmanDemo")
	cv.setMouseCallback('KalmanDemo', update_cursor_pos)

	# while True:
	    # state = 0.1 * np.random.randn(2, 1)

	print('--- transition ')
	print(kalman.transitionMatrix.shape)
	print(kalman.transitionMatrix)
	# kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
	kalman.transitionMatrix = np.array([[1.,0.,.3,0.],[0.,1.,0.,.3],[0.,0.,1.,0.],[0.,0.,0.,1.]])
	print(kalman.transitionMatrix.shape)
	print(kalman.transitionMatrix)
	print(kalman.transitionMatrix.dtype)

	print('---measurment matrix')

	print(kalman.measurementMatrix.shape)
	print(kalman.measurementMatrix)
	# kalman.measurementMatrix = cv.setIdentity(kalman.measurementMatrix, 1.)
	kalman.measurementMatrix = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])

	# kalman.measurementMatrix.astype(np.float64)
	print(kalman.measurementMatrix.shape)
	print(kalman.measurementMatrix)
	print(kalman.measurementMatrix.dtype)

	print('--- process noise cov')

	print(kalman.processNoiseCov.shape)
	print(kalman.processNoiseCov)
	kalman.processNoiseCov = 1e-4 * np.eye(4)
	# kalman.processNoiseCov = cv.setIdentity(kalman.processNoiseCov, 1e-4)
	print(kalman.processNoiseCov.shape)
	print(kalman.processNoiseCov)
	print(kalman.processNoiseCov.dtype)

	print('--- measurement noise cov')

	print(kalman.measurementNoiseCov.shape)
	print(kalman.measurementNoiseCov)
	kalman.measurementNoiseCov = 10 * np.eye(2)
	print(kalman.measurementNoiseCov.shape)
	print(kalman.measurementNoiseCov)
	print(kalman.measurementNoiseCov.dtype)

	print('--- error cov post')

	print(kalman.errorCovPost.shape)
	print(kalman.errorCovPost)
	kalman.errorCovPost = 1. * np.eye(4)
	print(kalman.errorCovPost.shape)
	print(kalman.errorCovPost)
	print(kalman.errorCovPost.dtype)

	print('--- state post')

	x = cursor_x
	y = cursor_y

	print(kalman.statePost.shape)
	print(kalman.statePost)
	kalman.statePost = 1. * np.array([[x],[y],[0.],[0.]])
	print(kalman.statePost.shape)
	print(kalman.statePost)
	print(kalman.statePost.dtype)

	img = np.zeros((img_height, img_width, 3), np.uint8)

	while(True):
		
		measurement = 1. * np.array([[0.],[0.]])
	
		global cursor_x, cursor_y

		measured_x = cursor_x
		measured_y = cursor_y
		

		measured_x = measured_x + np.random.randn(1, 1) * kalman.measurementNoiseCov[0,0]
		measured_y = measured_y + np.random.randn(1, 1) * kalman.measurementNoiseCov[1,1]

		measurement[0] = 1. * measured_x 
		measurement[1] = 1. * measured_y


		# print("measurement")
		# print(measurement)
		# print(measurement.dtype)

		# measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)

		global start
		if(start):
			cv.circle(img, (int(measurement[0]) , int(measurement[1])),2, (255, 255, 0), thickness=-1) 

		prediction = kalman.predict()

		# print("prediction")
		# print(prediction)

		# print('measurement matrix')
		# print(kalman.measurementMatrix)

		kalman.correct(measurement)

		if(start):
			cv.circle(img, (int(kalman.statePost[0]) , int(kalman.statePost[1])),2, (0, 255, 255), thickness=-1) 

		# print("corrected")
		# print(kalman.statePost)

		cv.imshow("KalmanDemo", img)

		code = cv.waitKey(10)
		if code != -1:
			break

    # point predictPt(prediction.at<float>(0),prediction.at<float>(1));



        # while True:
            # def calc_point(angle):
            #     return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
            #             np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

            # state_angle = state[0, 0]
            # state_pt = calc_point(state_angle)

            # prediction = kalman.predict()
        #     predict_angle = prediction[0, 0]
        #     predict_pt = calc_point(predict_angle)

        #     measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)

        #     # generate measurement
        #     measurement = np.dot(kalman.measurementMatrix, state) + measurement

        #     measurement_angle = measurement[0, 0]
        #     measurement_pt = calc_point(measurement_angle)

        #     # plot points
        #     def draw_cross(center, color, d):
        #         cv.line(img,
        #                  (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
        #                  color, 1, cv.LINE_AA, 0)
        #         cv.line(img,
        #                  (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
        #                  color, 1, cv.LINE_AA, 0)

        #     img = np.zeros((img_height, img_width, 3), np.uint8)
        #     draw_cross(np.int32(state_pt), (255, 255, 255), 3)
        #     draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
        #     draw_cross(np.int32(predict_pt), (0, 255, 0), 3)

        #     cv.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv.LINE_AA, 0)
        #     cv.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv.LINE_AA, 0)

        #     kalman.correct(measurement)

        #     process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(2, 1)
        #     state = np.dot(kalman.transitionMatrix, state) + process_noise

        #     cv.imshow("Kalman", img)

        #     code = cv.waitKey(100)
        #     if code != -1:
        #         break

        # if code in [27, ord('q'), ord('Q')]:
        #     break

	cv.destroyWindow("Kalman")
