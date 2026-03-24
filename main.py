import cv2
import numpy as np

camera = cv2.VideoCapture('test_video.mp4')

#WIDTH, HEIGHT = 240, 135
#WIDTH, HEIGHT = 1280, 720
WIDTH, HEIGHT = 640, 360

X_LIMIT=10**8
EPS=1e-9
def is_good_x(x):
    return -X_LIMIT<=x<=X_LIMIT


left_top=left_bottom=right_top=right_bottom=(0,0)
while True:
    ret, frame = camera.read()
    if not ret:
        break
    #2 - shrink frame
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_original=frame.copy()
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

    #9 remove edges + find white pixels coordinates(lines)
    frame_no_edges = frame.copy()

    edge_cols = int(frame.shape[1]* 0.05) # lungimea la [1], inaltime la [0]
    frame_no_edges[:,:edge_cols]=0
    frame_no_edges[:,-edge_cols:]=0
    # cut bottom noise
    bottom_rows = int(frame.shape[0]* 0.05)
    frame_no_edges[-bottom_rows:,:]=0

    frame=frame_no_edges

    #split in half
    MID = WIDTH // 2

    left_half = frame[:, :MID]
    right_half = frame[:, MID:]

    # detect line points
    left_side_points = np.argwhere(left_half > 0)
    right_side_points = np.argwhere(right_half > 0)

    #convert to global coordinates
    left_ys=left_side_points[:, 0]
    left_xs=left_side_points[:, 1]

    right_ys = right_side_points[:, 0]
    right_xs = right_side_points[:, 1] + MID

    #10
    b_left, a_left = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)   #y=ax+b
    b_right, a_right = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    y_top=0
    y_bottom=HEIGHT

    #left line: y=a_left*x+b_left-> x=(y - b_left)/a_left
    if abs(a_left) > EPS:
        x_top_left = int((y_top - b_left) / a_left)
        if is_good_x(x_top_left):
            left_top = (x_top_left, y_top)

        x_bottom_left = int((y_bottom - b_left) / a_left)
        if is_good_x(x_bottom_left):
            left_bottom = (x_bottom_left, y_bottom)

    #right line y=a_right*x+b_right-> x=(y - b_right)/a_right
    if abs(a_right) > EPS:
        x_top_right = int((y_top - b_right) / a_right)
        if is_good_x(x_top_right):
             right_top = (x_top_right, y_top)

        x_bottom_right = int((y_bottom - b_right) / a_right)
        if is_good_x(x_bottom_right):
             right_bottom = (x_bottom_right, y_bottom)

    #draw line
    frame_lines = frame.copy()
    cv2.line(frame_lines, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(frame_lines, right_top, right_bottom, (100, 0, 0), 5)

    # middle line
    cv2.line(frame_lines, (MID, 0), (MID, HEIGHT), (255, 0, 0), 1)


    #11- draw lines on original frame
    reverse_matrix = cv2.getPerspectiveTransform(frame_bounds, trapezoid_bounds)

    #left line on blank frame
    blank_frame_w_left_line = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.line(blank_frame_w_left_line, left_top, left_bottom, (255, 0, 0), 3)
    left_back = cv2.warpPerspective(blank_frame_w_left_line, reverse_matrix, (WIDTH, HEIGHT))

    #(y, x) where left line exists
    left_coords=np.argwhere(left_back > 0)

    #right line on blank frame
    blank_frame_w_right_line = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.line(blank_frame_w_right_line, right_top, right_bottom, (255, 0, 0), 3)
    right_back = cv2.warpPerspective(blank_frame_w_right_line, reverse_matrix, (WIDTH, HEIGHT))

    #(y, x) where right line exists
    right_coords=np.argwhere(right_back > 0)
    final_frame=frame_original.copy()

    # left line
    final_frame[left_coords[:, 0],left_coords[:, 1]]=(50, 50, 250)

    # right line
    final_frame[right_coords[:, 0],right_coords[:, 1]]=(50, 250, 50)
    frame=final_frame.copy()





    #cv2.imshow('frame_grey', frame_grey)
    #cv2.imshow('frame_birds_eye', frame_birds_eye)
    #cv2.imshow('frmae_blurred', frame_blurred)
    #cv2.imshow("frame_trapezoid", frame_trapezoid)
    #cv2.imshow('sabel_vertical',frame_sobel_vertical)
    #cv2.imshow('sabel_horizontal',frame_sobel_orizontal)
    cv2.imshow('left_line', left_back)
    cv2.imshow('right_line', right_back)
    cv2.imshow('frame', frame_lines)
    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()