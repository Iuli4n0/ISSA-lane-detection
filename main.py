import cv2
import numpy as np

camera = cv2.VideoCapture('test_video.mp4')

WIDTH, HEIGHT = 240, 135

while True:
    ret, frame = camera.read()
    if not ret:
        break
    #2 - shrink frame
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    #3 - convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_grey=frame.copy()

    #4 - select only the road
    x= int(WIDTH * 0.47)
    y = int(HEIGHT * 0.75)
    upper_left=(x,y)

    x= int(WIDTH * 0.53)
    y = int(HEIGHT * 0.75)
    upper_right=(x,y)

    x= int(0)
    y = int(HEIGHT)
    lower_left=(x,y)

    x= int(WIDTH)
    y = int(HEIGHT)
    lower_right=(x,y)

    trapezoid=np.array([upper_left, upper_right, lower_right, lower_left], dtype = np.int32)
    new_frame= np.zeros(frame.shape, dtype = np.uint8)
    cv2.fillConvexPoly(new_frame, trapezoid, 255)

    new_frame=new_frame//255 #transformam din 0/255 in 0/1
    frame = frame*new_frame
    frame_trapezoid=frame

    #5 birds-eye view
    trapezoid_bounds = np.float32(trapezoid)
    frame_bounds = np.float32([[0, 0], [WIDTH, 0],[WIDTH, HEIGHT],[0, HEIGHT]])

    magical_matrix=cv2.getPerspectiveTransform(trapezoid_bounds, frame_bounds)
    frame=cv2.warpPerspective(frame, magical_matrix, (WIDTH, HEIGHT))
    frame_birds_eye=frame.copy()


    #6 add blur
    frame= cv2.blur(frame, ksize = (5, 5))
    frame_blurred=frame.copy()

    #7 sobel filter
    sobel_vertical=np.float32([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_horizontal=np.transpose(sobel_vertical)

    copy1=frame.copy()
    copy1=np.float32(copy1)
    copy2=frame.copy()
    copy2=np.float32(copy2)

    copy1=cv2.filter2D(copy1, -1, sobel_vertical)
    copy2=cv2.filter2D(copy2, -1, sobel_horizontal)
    frame_sobel_vertical=cv2.convertScaleAbs(copy1)
    frame_sobel_orizontal=cv2.convertScaleAbs(copy2)

    frame=np.sqrt(np.square(copy1)+np.square(copy2))
    frame=cv2.convertScaleAbs(frame)

    #8 binarize the frame
    _, frame = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)

    #9

    cv2.imshow('frame_grey', frame_grey)
    cv2.imshow('frame_birds_eye', frame_birds_eye)
    cv2.imshow('frmae_blurred', frame_blurred)
    cv2.imshow("frame_trapezoid", frame_trapezoid)
    cv2.imshow('sabel_vertical',frame_sobel_vertical)
    cv2.imshow('sabel_horizontal',frame_sobel_orizontal)
    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()