import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from numpy.linalg import inv
from numpy.linalg import norm
import glob

class calibration(object):

    def __init__(self, path):
        # step 0: Camera Calibration and compute perspective matrix
        self.calibration(path)
        self.perspective_matrix()

    def calibration(self, path):
        objpoints = []
        imgpoints = []
        nx = 9
        ny = 6
        fnames = glob.glob(path)
        # forming object points like (0,0,0), (1,0,0)
        objp = np.zeros((nx*ny,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        #iteration
        for fname in fnames:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #find the corners
            ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)

            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
        self.mtx = mtx
        self.dist = dist

    def perspective_matrix(self):

        s1 = [207,720]
        s2 = [586,455]
        s3 = [698,455]
        s4 = [1127,720]
        side_offset = 160
        src = np.float32([s1,s2,s3,s4])
        dst = np.float32([[src[0,0]+side_offset,720],[src[0,0]+side_offset,0],[src[3,0]-side_offset,0],[src[3,0]-side_offset,720]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = inv(self.M)

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_fitx = []
        #average x values of the fitted line over the last n iterations
        self.best_fitx = None
        # current fitx
        self.current_fitx = None
        #polynomial coefficients the last n fits of the line
        self.recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #difference in fit coefficients between last and new fits
        self.diff_fit = None

        #radius of curvature of the line in some units
        self.curvature = None
        #line position
        self.pos = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # number of bad frames
        self.bad_ctr = 0

        # Previous fit coefficients (bad or good are all stored)
        self.previous_fit = None

class pipeline(object):

    def __init__(self, cali, left, right):
        self.mtx = cali.mtx
        self.dist = cali.dist
        self.M = cali.M
        self.Minv = cali.Minv
        self.left = left
        self.right = right
        self.reset = True
        self.bad_ctr = 0

    #def binary_image(self, img, s_thresh=(130, 255), sx_thresh=(25, 255)):
    def binary_image(self, img, s_thresh=(130, 255), sx_thresh=(35, 150)):
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def sliding_windows(self, img):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        binary_warped = np.copy(img)
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # -----------------Visualization
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        #Draw line
        for i in range(-3,4):
            out_img[np.int_(ploty), np.int_(left_fitx+i)] = [255, 255, 0]
        for i in range(-3,4):
            out_img[np.int_(ploty), np.int_(right_fitx+i)] = [255, 255, 0]

        self.left.allx = leftx
        self.left.ally = lefty
        self.left.current_fitx = left_fitx
        self.left.current_fit = left_fit
        self.right.allx = rightx
        self.right.ally = righty
        self.right.current_fitx  = right_fitx
        self.right.current_fit = right_fit

        return out_img

    def skip_windows(self, img):

        margin_left = 100
        margin_right = 100
        if self.left.detected is True:
            left_fit = self.left.recent_fit[-1]
        else:
            left_fit = self.left.best_fit
        if self.right.detected is True:
            right_fit = self.right.recent_fit[-1]
        else:
            right_fit = self.right.best_fit
        # left_fit = self.left.recent_fit[-1]
        # right_fit = self.right.recent_fit[-1]

        binary_warped = np.copy(img)
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Generate x and y values for previous windows
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx_win = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx_win = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx_win-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx_win+margin,
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx_win-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx_win+margin,
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        #Draw line
        for i in range(-3,4):
            idx = np.maximum(np.minimum(left_fitx+i,1279),0)
            result[np.int_(ploty), np.int_(idx)] = [255, 255, 0]
        for i in range(-3,4):
            idx = np.maximum(np.minimum(right_fitx+i,1279),0)
            result[np.int_(ploty), np.int_(idx)] = [255, 255, 0]

        self.left.allx = leftx
        self.left.ally = lefty
        self.left.current_fitx = left_fitx
        self.left.current_fit = left_fit
        self.right.allx = rightx
        self.right.ally = righty
        self.right.current_fitx  = right_fitx
        self.right.current_fit = right_fit

        return result

    def measurement(self):

        lefty = self.left.ally
        leftx = self.left.allx
        righty = self.right.ally
        rightx = self.right.allx

        left_fitx = self.left.current_fitx
        right_fitx = self.right.current_fitx


        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3/70 # meters per pixel in y dimension
        xm_per_pix = 3.7/623 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        y_eval_m = 719 * ym_per_pix

        diff_m =  (right_fitx - left_fitx)*xm_per_pix

        self.left.curvature = ((1 + (2*left_fit_cr[0]*y_eval_m + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right.curvature = ((1 + (2*right_fit_cr[0]*y_eval_m + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        self.left.pos = left_fit_cr[0]*y_eval_m**2 + left_fit_cr[1]*y_eval_m + left_fit_cr[2]
        self.right.pos = right_fit_cr[0]*y_eval_m**2 + right_fit_cr[1]*y_eval_m + right_fit_cr[2]
        self.line_var = np.var(diff_m)
        self.line_mean = np.mean(diff_m)
        self.line_ratio = np.abs(np.log10(self.left.curvature/self.right.curvature))

    def draw(self, img):

        lefty = self.left.ally
        leftx = self.left.allx
        righty = self.right.ally
        rightx = self.right.allx
        Minv = self.Minv
        left_curverad = self.left.curvature
        right_curverad = self.right.curvature
        left_pos = self.left.pos
        right_pos = self.right.pos
        line_var = self.line_var
        line_mean = self.line_mean


        # if self.valid is True:
        #     left_fitx = self.left.current_fitx
        #     right_fitx = self.right.current_fitx
        # else:
        #     left_fitx = self.left.best_fitx
        #     right_fitx = self.right.best_fitx
        left_fitx = self.left.best_fitx
        right_fitx = self.right.best_fitx

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
        img_draw = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        # Draw the points detected
        left_warp = np.zeros_like(img[:,:,0]).astype(np.uint8)
        right_warp = np.zeros_like(img[:,:,0]).astype(np.uint8)
        left_warp[lefty, leftx] = 1
        right_warp[righty, rightx] = 1
        left_unwwarp = cv2.warpPerspective(left_warp, Minv, (left_warp.shape[1], left_warp.shape[0]))
        right_unwwarp = cv2.warpPerspective(right_warp, Minv, (right_warp.shape[1], right_warp.shape[0]))
        left_unwwarp.nonzero()
        img_draw[left_unwwarp.nonzero()[0],left_unwwarp.nonzero()[1],:] = np.int_(0.5*np.array(img_draw[left_unwwarp.nonzero()[0],left_unwwarp.nonzero()[1],:])+ [128,0,0])
        img_draw[right_unwwarp.nonzero()[0],right_unwwarp.nonzero()[1],:] = np.int_(0.3*np.array(img_draw[right_unwwarp.nonzero()[0],right_unwwarp.nonzero()[1],:])+ [0,0,178])


        #Putting Text
        xm_per_pix = 3.7/623 # meters per pixel in x dimension

        mean_curverad = (left_curverad+right_curverad)/2
        mean_pos = (left_pos+right_pos)/2
        offset_m = 3.5716 - mean_pos
        offset_dir = 'right' if offset_m > 0 else 'left'

#         cv2.putText(img_draw, text="Radius = "+str(int(mean_curverad)) + "(m)", org=(50,50),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img_draw, text='Curvature: {0:} m'.format(int(mean_curverad)), org=(50,50),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img_draw, text='Vheicle is {0:.2g}m {1} of center'.format(offset_m,offset_dir), org=(50,100),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)
        cv2.putText(img_draw, text='var: {0:.5f}, mean: {1:4f}'.format(line_var,line_mean), org=(50,150),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)
        # cv2.putText(img_draw, text='lf = {0:.3f}, {1:.3f}, {2:.3f}'.format(self.left.diff_fit[0],self.left.diff_fit[1],self.left.diff_fit[2]), org=(50,200),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)
        # cv2.putText(img_draw, text='rt = {0:.3f}, {1:.3f}, {2:.3f}'.format(self.right.diff_fit[0],self.right.diff_fit[1],self.right.diff_fit[2]), org=(50,250),fontFace=2, fontScale=1, color=(255,255,255), thickness=2)

        # if self.valid is False:
        #     cv2.putText(img_draw, text='False', org=(50,300),fontFace=2, fontScale=1, color=(255,0,0), thickness=2)
        if self.left.detected is True:
             cv2.putText(img_draw, text='True', org=(50,200),fontFace=2, fontScale=1, color=(0,255,0), thickness=2)
        else:
             cv2.putText(img_draw, text='False', org=(50,200),fontFace=2, fontScale=1, color=(255,0,0), thickness=2)
        if self.right.detected is True:
             cv2.putText(img_draw, text='True', org=(150,200),fontFace=2, fontScale=1, color=(0,255,0), thickness=2)
        else:
             cv2.putText(img_draw, text='False', org=(150,200),fontFace=2, fontScale=1, color=(255,0,0), thickness=2)

        return img_draw

    def draw_optional(self, img1, img2):
        img_large = np.copy(img1)
        img_small = cv2.resize(img2,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        img_large[:img_small.shape[0],-img_small.shape[1]:] = img_small

        return img_large

    def validation(self):
        # if self.line_var < 0.2 and self.line_ratio < 1.5:
        #     self.valid = True
        self.valid = True

    def update(self, obj):
        n = 8
        if self.reset is True:
            obj.recent_fit = []
            obj.recent_fitx = []
            obj.diff_fit = np.array([0.0,0.0,0.0])
            obj.bad_ctr = 0
        else:
            if obj.detected is True:
                obj.diff_fit = np.abs(obj.current_fit - obj.previous_fit)*[8000,2.5,1/80]
            else:
                obj.diff_fit = np.abs(obj.current_fit - obj.previous_fit)*[8000,2.5,1/80]*4


        if np.all(obj.diff_fit < np.array([1,1,1])):
            obj.recent_fitx.append(obj.current_fitx)
            if len(obj.recent_fitx) > n:
                obj.recent_fitx.pop(0)
            obj.best_fitx = np.mean(obj.recent_fitx,0)

            obj.recent_fit.append(obj.current_fit)
            if len(obj.recent_fit) > n:
                obj.recent_fit.pop(0)
            obj.best_fit = np.mean(obj.recent_fit,0)

            obj.detected = True
            obj.bad_ctr = 0
        else:
            obj.detected = False
            obj.bad_ctr += 1
        obj.previous_fit = obj.current_fit


    def process_image(self, img):
        self.valid = False
        img_size = (img.shape[1],img.shape[0])
        #Step 1: undistort image
        img_undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        #Step 2: Generate binary image
        img_binary = self.binary_image(img_undist)

        #Step 3: Perspective Transfrom
        img_warp = cv2.warpPerspective(img_binary, self.M, img_size, flags=cv2.INTER_NEAREST)

        #Step 4: line searching
        if self.reset is True:
            img_windows = self.sliding_windows(img_warp)
            self.valid = True
        else:
            img_windows = self.skip_windows(img_warp)

        #Step 5: compute curvature, position,...etc
        self.measurement()

        #Step 6: Validity Check
        self.validation()

        self.update(self.left)
        self.update(self.right)

        # number of bad frames before reset for new window researching
        n = 10
        if self.left.bad_ctr == n or self.right.bad_ctr == n:
            self.reset = True
        else:
            self.reset = False

        #Step 7: Drawing
        img_draw = self.draw(img_undist)
        # img_draw = self.draw_optional(img_draw,img_windows)

        return img_draw
