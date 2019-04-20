import cv2

cap = cv2.VideoCapture(0)

cv2.namedWindow("Test")
cv2.resizeWindow("Test", 300, 150)

img_counter = 0

while True:
	# capture frame-by-frame
	ret, frame = cap.read()
	# frame = cv2.flip(frame,1)
	frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)

	# operations on the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# display resulting frame
	# cv2.imshow('Capturing', gray)

	key = cv2.waitKey(1)

	if key%256 == 27:
		# esc pressed
		print("Escape pressed, closing...")
		break
	elif key%256 == 32:
		# space pressed
		img_name = "frame_{}.png".format(img_counter)
		cv2.imwrite(img_name, gray)
		print("{} written!".format(img_name))
		print(gray)
		img_counter += 1

# release the capture
cap.release()
cv2.destroyAllWindows()