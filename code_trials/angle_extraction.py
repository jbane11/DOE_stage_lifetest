import pathlib, datetime, atexit
import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy,os
from scipy.optimize import minimize
# from typing import Tuple, List, Optional
# import math
from sklearn.linear_model import RANSACRegressor
# from sklearn.cluster import KMeans
import numpy as np

import json, logging

import logging.handlers
import logging.config

logger = logging.getLogger("DOE_log")
config_file = pathlib.Path("logging_config.json")

#grab the current path of this script
current_path = pathlib.Path(__file__).parent
if not config_file.exists():
    config_file = current_path / "logging_config.json"

with open(config_file) as f_in:
    config = json.load(f_in)

today = datetime.date.today().strftime("%Y%m%d")

config["handlers"]["stderr"]["filename"] = f"logs/{today}-stderr.log"
config["handlers"]["file_json"]["filename"] = f"logs/{today}-Lifetest.log"
config["handlers"]["DEBUG"]["filename"] = f"logs/{today}-debug.log"

logging.config.dictConfig(config)
logging.getLogger('matplotlib.font_manager').disabled = True
queue_handler = logging.getHandlerByName("queue_handler")
if queue_handler is not None:
    queue_handler.listener.start()
    atexit.register(queue_handler.listener.stop)

logger.info("Logging setup complete.")





def horizontal_edges(image, resolution:int=1, plot=0, verbose=False)->np.ndarray:
    ''' Scans an image horizontally to find edges using peak detection.
        Args:
        image (np.ndarray): Input image, can be grayscale or color.
        resolution (int): Step size for scanning the image vertically. Default is 1.
        plot (int): Level of plotting detail. 0 = no plots, 1 = final plot only, 2 all plots,
        >2 = every n-1 *10 plots. Default is 0.
        2 = all plots, >2 = every nth plot.
        verbose (bool): If True, prints detailed information during processing. Default is False. 

        returns:
        np.ndarray: Array of detected edge points with shape (N, 2), where each point is (x, y).
        '''
    #check to see if image is color or grayscale
    if len(image.shape)==3 and image.shape[2]==3:
        image_rgb = image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    H_limit_hard = [450,1095]
    V_limit_hard = [425,950]

    Vertical_limits = [400,1050]
    Horizontal_limits = [400,1200]

    if plot >=1:
        fig1, ax1 = plt.subplots(figsize=(10,8))
        ax1.imshow(image_rgb)


    circle_peaks = []

    for y in range(V_limit_hard[0],V_limit_hard[1],resolution):

        
        full_raw_line = max(gray_image[y,:]) - gray_image[y,:]
        
        line= full_raw_line.copy()
        line[:Horizontal_limits[0]] = full_raw_line[Horizontal_limits[0]]
        line[Horizontal_limits[1]:] = full_raw_line[Horizontal_limits[1]-1]
        line_blurred = line #np.array(cv2.GaussianBlur(line, (5, 5), 0)[:,0])
        
        height = np.mean(line[Horizontal_limits[0]:Horizontal_limits[1]])*1.225


        peaks,peak_properties = scipy.signal.find_peaks(line_blurred,width=[1,9.3],
                                                     prominence=12.5,height=height,
                                                     distance=40)# Filter peaks to find pairs that correspond to circle edges

        #take the two extreme peaks as the circle edges
        if len(peaks)>=2:
            peaks=[min(peaks), max(peaks)]

        for peak in peaks:
            if plot==2: 
                fig,ax = plt.subplots(figsize=(10,8))
                ax.plot(full_raw_line)
                ax.plot(line_blurred)
                ax.plot(peak, line_blurred[peak], "x", markersize=10, color='red')
                ax.grid()
                ax.set_title(f"y={y}")
            elif plot>2:
                if y%((plot-1)*10)==0:
                    fig,ax = plt.subplots(figsize=(10,8))
                    ax.plot(peak, line_blurred[peak], "x", markersize=10, color='red')
                    ax.plot(full_raw_line)
                    ax.plot(line_blurred)
                    ax.plot(peak, line_blurred[peak], "x", markersize=10, color='red')
                    ax.grid()
                    ax.set_title(f"y={y}")
            
            
            if plot>=1: 
                ax1.plot(peak, y, "x", color='yellow')
            

            circle_peaks.append((peak,y))
        if verbose:
            print(f"Found peaks at y={y}:")
            print(peaks)
            for key in peak_properties.keys():
                print(f"{key}: {peak_properties[key]}")
    return np.array(circle_peaks)   


def Center_Radius(peaks, plot=True, verbose=True):
    ''' Fits a circle to a set of 2D points using RANSAC to filter out outliers.
        Args:
        peaks (np.ndarray): Array of shape (N, 2) containing the (x, y) coordinates of the points.
        plot (bool): If True, plots the points and the fitted circle. Default is True.
        verbose (bool): If True, prints detailed information about the fitting process. Default is True.
        
        Returns:
        Tuple containing:
            - xc (float): x-coordinate of the circle center.
            - yc (float): y-coordinate of the circle center.
            - R (float): Radius of the fitted circle.
            - std_R (float): Standard deviation of the radius.
            - res (float): Residual error of the fit. '''


    # fit a circle to the inlier points
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((inlier_points[:,0]-xc)**2 + (inlier_points[:,1]-yc)**2) 

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        residuals = Ri - Ri.mean()
        return np.sum(residuals**2)  # Return sum of squared residuals (scalar)

    # find the best fit circle to the points, some points are bad, use RANSAC to filter out the bad points
    #reformat the data
    circle_peaks = np.array(peaks)
    X = circle_peaks[:,0].reshape(-1,1)
    Y = circle_peaks[:,1].reshape(-1,1)
    data = np.hstack((X,Y))
    ransac = RANSACRegressor(min_samples=int(len(data)*0.30), 
                            residual_threshold=35, max_trials=3000)
    ransac.fit(X, Y) # fit the model
    inlier_mask = ransac.inlier_mask_  # get the inlier mask
    outlier_mask = np.logical_not(inlier_mask)  
    inlier_points = data[inlier_mask]
    outlier_points = data[outlier_mask]
    center_estimate = np.mean(inlier_points, axis=0)
    center_2 = minimize(f_2, center_estimate).x
    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)         
    R_2        = Ri_2.mean()
    res_2      = np.sum((Ri_2 - R_2)**2)        




    if verbose:
        print(f"Total points: {len(circle_peaks)}")
        print(f"RANSAC found {len(inlier_points)} inliers and {len(outlier_points)} outliers.")
        print(f"Circle center: ({xc_2:.2f}, {yc_2:.2f})")
        print(f"Circle radius: {R_2:.2f} +/- {np.std(Ri_2):.2f}")
        print(f"residual: {res_2:.2f}") 

    
    if plot:
        fig,ax = plt.subplots(figsize=(10,8))
        ax.plot(circle_peaks[:,0], circle_peaks[:,1], 'b.', label='All Detected Peaks')
        ax.plot(outlier_points[:,0], outlier_points[:,1], 'ro', label='outliers',alpha=0.7)
        ax.plot(inlier_points[:,0], inlier_points[:,1], 'go', label='Inliers')
        ax.plot(xc_2, yc_2, 'yx', markersize=10, label='Fitted Circle Center')

        circle = plt.Circle((xc_2, yc_2), R_2, color='r', fill=False, linewidth=2)
        ax.add_patch(circle)   
        ax.set_title('RANSAC Circle Fit')
        #ax.set_aspect('equal', adjustable='box')
        ax.grid()
        plt.legend()

    return (xc_2, yc_2, R_2, np.std(Ri_2), res_2)


def Center_Radius_iterations(peaks=None, max_iterations=10, threshold=0.02, plot=True, verbose=True):

    # --- Circle fit helper functions ---
    def algebraic_circle_fit(x, y):
        """Fit circle to points using algebraic least squares."""
        D = np.column_stack([x, y, np.ones_like(x)])
        b = -(x**2 + y**2)
        sol, *_ = np.linalg.lstsq(D, b, rcond=None)
        A, B, C = sol
        cx = -A / 2
        cy = -B / 2
        r = np.sqrt(cx**2 + cy**2 - C)
        return cx, cy, r

    def fit_and_residuals(pts):
        """Fit a circle and return center, radius, and residuals for each point."""
        x, y = pts[:,0], pts[:,1]
        cx, cy, r = algebraic_circle_fit(x, y)
        dists = np.sqrt((x - cx)**2 + (y - cy)**2)
        res = dists - r
        return cx, cy, r, res




    # --- Iterative outlier removal ---
    remaining = peaks.copy()
    removed = []
    prev_len = None
    iteration = 0




    while True:
        iteration += 1
        cx, cy, r, res = fit_and_residuals(remaining)
        rms = np.sqrt(np.mean(res**2))

        # Median & MAD (robust statistics)
        med = np.median(res)
        mad = np.median(np.abs(res - med))
        sigma_est = 1.4826 * mad if mad > 0 else np.std(res)

        # Outlier threshold
        thresh = max(2.5 * sigma_est, 0.02 * r)
        outlier_mask = np.abs(res - med) > thresh

        if outlier_mask.sum() == 0:
            break

        removed.extend(remaining[outlier_mask].tolist())
        remaining = remaining[~outlier_mask]

        if prev_len == len(remaining) or len(remaining) < 4 or iteration > max_iterations:
            break
        prev_len = len(remaining)

    # --- Final fit on cleaned data ---
    final_cx, final_cy, final_r, final_res = fit_and_residuals(remaining)
    final_rms = np.sqrt(np.mean(final_res**2))
    if verbose:
        print("Final fit:")
        print(f" Center: ({final_cx:.6f}, {final_cy:.6f})")
        print(f" Radius: {final_r:.6f}")
        print(f" RMS residual: {final_rms:.6f}")
        print(f" Points kept: {len(remaining)}, removed: {len(removed)}")

    if removed:
        rem_arr = np.array(removed)

    if plot:
        # --- Plot result ---
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(remaining[:,0], remaining[:,1], label='kept points')
        if removed:
            ax.scatter(rem_arr[:,0], rem_arr[:,1], c='red', marker='x', label='removed points')

        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(final_cx + final_r*np.cos(theta), final_cy + final_r*np.sin(theta), 'k-')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Final circle fit with outlier removal')
        ax.legend()
        ax.invert_yaxis()  # Invert y-axis for image coordinates
        ax.grid()
        plt.show()


    return final_cx, final_cy, final_r, final_rms, remaining, removed




def horizontal_scan_for_center_peaks(image, resolution=50, center_info=None, plot=1, verbose=False):
    
    #check if image is color or grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    if center_info is None:
        if verbose: print("No center info provided.")
        edges=horizontal_edges(gray_image, resolution=10, plot=0, verbose=False)
        center_info =Center_Radius_iterations(edges, plot=False, verbose=False)

    cx, cy, r, rms = center_info[:4]
    inner_circle_rad=   r-100
    H_limit = [int(cx-inner_circle_rad), int(cx+inner_circle_rad)]
    V_limit = [int(cy-inner_circle_rad), int(cy+inner_circle_rad)]

    Horizontal_limits = [int(cx-r),int(cx+r)]
    Vertical_limits = [int(cy-r),int(cy+r)]
    
    

    if plot:
      fig,ax = plt.subplots(figsize=(10,4))
      ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      ax.plot(cx, cy, 'ro', label='Circle Center')
      inner_circle = plt.Circle((cx, cy), inner_circle_rad, color='blue',
                          fill=False, linestyle='--', linewidth=1, label='Inner Boundary')
      ax.add_artist(inner_circle) 
      ax.grid()


    All_peaks = []
    # start scanning lines
    for j,y in enumerate(range(V_limit[0],V_limit[1],resolution)):

        if verbose:
            print(f"Line {j} at y={y}")

        full_scan = np.max(gray_image[y, :])-gray_image[y,:]

        # calculate left and right limits for the line scan useing circle equation

        square_diff =(np.ceil(inner_circle_rad**2) - (np.floor(y - cy))**2)
        mod = 0
        while square_diff <= 0:
            mod += 1
            if verbose:
                print(f"Line {j} at y={y} - outside inner circle")
            y = int(cy - inner_circle_rad)+mod
            square_diff =(np.ceil(inner_circle_rad**2) - (np.floor(y - cy))**2)
            if verbose:   print(f"setting y to inside inner circle, {y}")
            if mod>5:
                break
                

        left_most = int(cx - np.sqrt(square_diff))
        right_most = int(cx + np.sqrt(square_diff))

        line_scan = full_scan.copy()
        left_value =line_scan[left_most+2]
        right_value=line_scan[right_most-2]
        endcap_avg = (float(left_value)+float(right_value))/2.0
        # print("left and right values",left_value,right_value)
        # print("endcap avg",endcap_avg)

        line_scan[:left_most]  = line_scan[left_most+2]
        line_scan[right_most:] = line_scan[right_most-2]
        # print("left and right most",left_most,right_most)




        #smoothing algorithm parameters
        # window_length must be odd and less than or equal to the size of line_scan

        line_scan_diff = np.gradient(full_scan )
        line_scan_diff[:left_most+20]  = 0
        line_scan_diff[right_most-20:] = 0

        # print("line scan diff ::10",line_scan_diff[::10])

        # print("max diff",np.max(line_scan_diff), " min diff", np.min(line_scan_diff))
        # print("Std dev", np.std(line_scan_diff))

        
        window_length = 41
        polyorder = 6

        if np.std(line_scan_diff) < 1.3:
            window_length = 11
            polyorder = 6
        elif np.std(line_scan_diff) < 2:
            window_length = 31
            polyorder = 5
        elif np.std(line_scan_diff) < 3:
            window_length = 35
            polyorder = 4
        elif np.std(line_scan_diff) >= 3:
            window_length = 41
            polyorder = 3

        


        smooth_scan = scipy.signal.savgol_filter(line_scan, 
                                    window_length=window_length,
                                    polyorder=polyorder)


        if plot>1:
            ax.hlines(y, H_limit[0], H_limit[1],
                    colors='red', linestyles='dashed', linewidth=1)

        
        if plot > 1:
            
            fig2,ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(line_scan, color="blue",label=f'Line {j} at y={y}')
            ax2.plot(smooth_scan, color="green",label='Smoothed Scan')
            ax2.grid() 
            ax2.legend()
            ax2.set_xlim(Horizontal_limits)

        peaks=[]

        peaks,peak_props = scipy.signal.find_peaks(smooth_scan,
                                                width=[3,200],
                                                prominence=15,
                                                    height=1)

        if verbose:
            print(f"  Found {len(peaks)} peaks at x-positions: {peaks}") 
            for key in peak_props:
                print(f"    {key}: {peak_props[key]}")  


        if len(peaks) > 0:
            if plot > 1:
                ax2.scatter(peaks, line_scan[peaks], color='red', s=50, label='Detected Peaks')
            if plot >= 1:    
                ax.scatter(peaks, np.full_like(peaks, y), color='cyan', s=20, label='Detected Peaks')

        #If more then two peaks are found pick the best two based on prominence
        if len(peaks) > 2:
            prominences = peak_props['prominences']
            top_two_indices = np.argsort(prominences)[-2:]
            peaks = peaks[top_two_indices]


        All_peaks.append(np.column_stack((peaks, np.full_like(peaks, y))))

    return np.vstack(All_peaks)

#veritcal scan function to find peaks
def vertical_scan_for_center_peaks(image, resolution=50, center_info=None, plot=1, verbose=False):
    
    #check if image is color or grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    if center_info is None:
        if verbose:print("No center info provided.")
        edges=horizontal_edges(gray_image, resolution=10, plot=0, verbose=False)
        center_info =Center_Radius_iterations(edges, plot=False, verbose=False)

    cx, cy, r, rms = center_info[:4]
    inner_circle_rad=   r-99

    H_limit = [int(cx-inner_circle_rad), int(cx+inner_circle_rad)]
    V_limit = [int(cy-inner_circle_rad), int(cy+inner_circle_rad)]

    Horizontal_limits = [int(cx-r),int(cx+r)]
    Vertical_limits = [int(cy-r),int(cy+r)]

    

    if plot:
      fig,ax = plt.subplots(figsize=(6,10))
      ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      ax.plot(cx, cy, 'ro', label='Circle Center')
      inner_circle = plt.Circle((cx, cy), inner_circle_rad, color='blue',
                          fill=False, linestyle='--', linewidth=1, label='Inner Boundary')
      ax.add_artist(inner_circle) 
      ax.grid()


    All_peaks = []
    # start scanning lines
    for j,x in enumerate(range(H_limit[0],H_limit[1], resolution)):
        # if j<22 or j>29:
        #     continue
        if verbose:
            print(f"Line {j} at x={x}")

        full_scan = np.max(gray_image[:, x])-gray_image[:,x]
        line_scan = full_scan.copy()

        # calculate top and bottom limits for the line scan useing circle equation

        square_diff =(np.ceil(inner_circle_rad**2) - (np.floor(x - cx))**2)
        if square_diff <= 0:
            if verbose:
                print(f"Skipping line {j} at x={x} - outside inner circle")
            continue

        top_most = int(cy - np.sqrt(square_diff))
        bottom_most = int(cy + np.sqrt(square_diff))

        top_value =line_scan[top_most+20]
        bottom_value=line_scan[bottom_most-20]
        endcap_avg = (float(top_value)+float(bottom_value))/2.0
        line_scan[:top_most+20]    = line_scan[top_most+20]
        line_scan[bottom_most-20:] = line_scan[bottom_most-20]


                #smoothing algorithm parameters
        # window_length must be odd and less than or equal to the size of line_scan

        line_scan_diff = np.gradient(full_scan )
        line_scan_diff[:top_most+20]  = 0
        line_scan_diff[bottom_most-20:] = 0

        # print("line scan diff ::10",line_scan_diff[::10])

        # print("max diff",np.max(line_scan_diff), " min diff", np.min(line_scan_diff))
        # print("Std dev", np.std(line_scan_diff))

        
        window_length = 41
        polyorder = 6

        if np.std(line_scan_diff) < 1.3:
            window_length = 11
            polyorder = 6
        elif np.std(line_scan_diff) < 2:
            window_length = 31
            polyorder = 5
        elif np.std(line_scan_diff) < 3:
            window_length = 35
            polyorder = 4
        elif np.std(line_scan_diff) >= 3:
            window_length = 41
            polyorder = 3

        smooth_scan = scipy.signal.savgol_filter(line_scan,
                            window_length=window_length,
                            polyorder=polyorder) 


        
        if plot>1:
            ax.vlines(x, V_limit[0], V_limit[1],
                    colors='red', linestyles='dashed', linewidth=1)

        
        if plot > 1:
            
            fig2,ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(line_scan, color="blue",label=f'Line {j} at x={x}')
            ax2.plot(smooth_scan, color="green",label='Smoothed Scan')
            ax2.vlines(top_most, 0, np.max(line_scan),
                    colors='red', linestyles='dashed', linewidth=1, label='top Limit')
            ax2.vlines(bottom_most, 0, np.max(line_scan),
                    colors='orange', linestyles='dashed', linewidth=1, label='bottom Limit')

            ax2.hlines(endcap_avg, 0, len(line_scan),
                    colors='black', linestyles='dashed', linewidth=1, label='End Cap Avg')

            ax2.grid() 
            ax2.legend()
            ax2.set_xlim(Vertical_limits)


        peaks=[]

        peaks,peak_props = scipy.signal.find_peaks(smooth_scan,
                                                width=[3,45],
                                                prominence=14.5,
                                                    height=1)

        # peaks1,peak_props1 = scipy.signal.find_peaks(smooth_scan,
        #                                         width=[1,85],
        #                                         prominence=4,
        #                                         threshold=1,
        #                                             height=1)

        if len(peaks)>0:
            if abs(peaks[0]-top_most)<20:
                if verbose:
                    print(f"  Removing peak at {peaks[0]} - too close to top limit {top_most}")
                peaks=peaks[1:]
                peak_props={key: peak_props[key][1:] for key in peak_props}  # Update peak_props accordingly
            if len(peaks)>0 and abs(peaks[-1]-bottom_most)<20:
                if verbose:
                    print(f"  Removing peak at {peaks[-1]} - too close to bottom limit {bottom_most}")
                peaks=peaks[:-1]
                peak_props={key: peak_props[key][:-1] for key in peak_props}  # Update peak_props accordingly
            # check for prominence/width ratio
            prominences = peak_props['prominences']
            widths = peak_props['widths']
            squat_ratio = prominences/widths
            # print(squat_ratio)
            #drop peaks that have a squat ratio < 1.3
            valid_peaks_mask = squat_ratio >= 1.2
            peaks = peaks[valid_peaks_mask]
            peak_props = {key: peak_props[key][valid_peaks_mask] for key in peak_props}




        if verbose:
            print(f"  Found {len(peaks)} peaks at x-positions: {peaks}") 
            for key in peak_props:
                print(f"    {key}: {peak_props[key]}")  
        logger.debug(f"Line {j} at x={x} - Found {len(peaks)} peaks at x-positions: {peaks}")
        for key in peak_props:
            logger.debug(f"    {key}: {peak_props[key]}")   

        # print(peaks1)
        # for key in peak_props1:
        #     print(f"    {key}: {peak_props1[key]}")


        if len(peaks) > 0:
            if plot > 1:
                ax2.scatter(peaks, line_scan[peaks], color='red', s=50, label='Detected Peaks')
            if plot >= 1:
                ax.scatter(np.full_like(peaks, x), peaks, color='cyan', s=20, label='Detected Peaks')
        # if len(peaks1) > 0:
        #     ax2.scatter(peaks1, smooth_scan[peaks1], color='red', s=50, label='All Detected Peaks')
            
        #If more then two peaks are found pick the best two based on prominence
        if len(peaks) > 2:
            prominences = peak_props['prominences']
            top_two_indices = np.argsort(prominences)[-2:]
            peaks = peaks[top_two_indices]


        All_peaks.append(np.column_stack((np.full_like(peaks, x), peaks)))

    return np.vstack(All_peaks)
            


def find_two_lines(data, plot=True, verbose=False):
    """
    Find two lines in the given 2D data using RANSAC algorithm.

    Parameters:
    data (numpy.ndarray): A 2D array of shape (N, 2) where each row represents a point (x, y).
    plot (bool): If True, plot the data points and the fitted lines.
    verbose (bool): If True, print additional information during processing.

    Returns:
    tuple: Coefficients of the two lines in the form (slope1, intercept1, slope2, intercept2).
    """
        
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import RANSACRegressor, LinearRegression




    points = data

    points = np.array(points)
    # radomly  sort the points
    np.random.shuffle(points)
    
    X = points[:,0].reshape(-1,1)
    y = points[:,1]

    # Use RANSAC to detect first line
    ransac1 = RANSACRegressor(LinearRegression(), residual_threshold=5, random_state=0)
    ransac1.fit(X, y)
    inlier_mask1 = ransac1.inlier_mask_
    outlier_mask1= np.logical_not(inlier_mask1)

    y_pred1 = ransac1.predict(X[inlier_mask1])

    #standard devoation of residuals
    residuals1 = np.abs(y[inlier_mask1] - y_pred1)

    Full_pred = ransac1.predict(X)

    Full_residuals = np.abs(y-Full_pred)



    # fit new line to data that has Full_residulas <2
    low_residuals_mask = Full_residuals < 2
    ranscac1_1 = RANSACRegressor(LinearRegression(), residual_threshold=15, random_state=10)
    ranscac1_1.fit(X[low_residuals_mask], y[low_residuals_mask])

    
    x1_1 = np.arange(X[low_residuals_mask].min(),X[low_residuals_mask].max(),0.1).reshape(-1,1)
    y1_1 = ranscac1_1.predict(x1_1) 
    if plot:
        fig, ax  = plt.subplots()
        plt.scatter(X[low_residuals_mask], y[low_residuals_mask],
                 marker="D",color="magenta", s=30, label="data_low_residuals")



    X_remaing = X[outlier_mask1]
    y_remaining = y[outlier_mask1]
    # Use RANSAC on remaining points for second line
    ransac2 = RANSACRegressor(LinearRegression(), residual_threshold=10, random_state=1)
    ransac2.fit(X_remaing, y_remaining)
    inlier_mask2 = ransac2.inlier_mask_

    y_pred2 = ransac2.predict(X_remaing[inlier_mask2])
    #standard devoation of residuals
    residuals2 = np.abs(y_remaining[inlier_mask2] - y_pred2)

    
    #slopes and intercepts
    slope1 = ransac1.estimator_.coef_[0]
    intercept1 = ransac1.estimator_.intercept_
    slope2 = ransac2.estimator_.coef_[0]
    intercept2 = ransac2.estimator_.intercept_

    line_x2 = np.arange(X_remaing[inlier_mask2].min(),X_remaing[inlier_mask2].max(),0.1)
    line_y2 = slope2 * line_x2 + intercept2

    line_x1 = np.arange(X[inlier_mask1].min(),X[inlier_mask1].max(),0.1)
    line_y1 = slope1 * line_x1 + intercept1
    # Plot

    #length of lines
    line1_length = np.sqrt((line_x1[-1]-line_x1[0])**2 + (line_y1[-1]-line_y1[0])**2)
    line2_length = np.sqrt((line_x2[-1]-line_x2[0])**2 + (line_y2[-1]-line_y2[0])**2)
    if verbose:
            print("Standard deviation of residuals for line 1:", np.std(residuals1))
            print("Number of inliers for line 1:", np.sum(inlier_mask1), "out of", len(y))
            print("Standard deviation of residuals for line 2:", np.std(residuals2))
            print("Number of inliers for line 2:", np.sum(inlier_mask2), "out of", len(y_remaining))
            print(f"Line 1 length: {line1_length}, Line 2 length: {line2_length}")
    flipped=0
    if line1_length < line2_length:
        flippped=1
        if verbose:         print("Warning: Line 1 is shorter than Line 2.")
        temp_slope = slope1
        slope1 = slope2
        slope2 = temp_slope
        temp_intercept = intercept1
        intercept1 = intercept2
        intercept2 = temp_intercept

        temp_line_x = line_x1
        line_x1 = line_x2   
        line_x2 = temp_line_x

        temp_line_y = line_y1
        line_y1 = line_y2
        line_y2 = temp_line_y

    if plot:
        # plot the first inliners and its prediction
        # fig, ax  = plt.subplots()
        plt.scatter(X[inlier_mask1], y[inlier_mask1],
                     color="gray",marker="d", s=20, label="data_line1",)
        plt.scatter(X[inlier_mask1], y_pred1, 
                    color="green",marker="+", s=20, label="line1 inliers")  

        plt.scatter(X_remaing[inlier_mask2], y_remaining[inlier_mask2],
                     color="slategray", marker="o", s=20, label="data_line2")
        plt.scatter(X_remaing[inlier_mask2], y_pred2, color="blue", marker="x", s=20, label="line2 inliers")

        final_remaing_x = X_remaing[~inlier_mask2]
        final_remaing_y = y_remaining[~inlier_mask2]

        plt.scatter(final_remaing_x, final_remaing_y, color="red", marker="s", s=20, label="line2 outliers")
        plt.plot(line_x1, line_y1, "g-", label="Line 1 fit")
        plt.plot(line_x2, line_y2, "b-", label="Line 2 fit")    
        ax.yaxis.set_inverted(True)
        plt.grid()
        plt.legend()
        plt.show()
    print(X[0], y[0])
    #x and y points for the two fits
    if flipped==1:
        points_2 = np.hstack((X[inlier_mask1], np.array(y[inlier_mask1]).reshape(-1,1)))
        points_1 = np.hstack((X_remaing[inlier_mask2], np.array(y_remaining[inlier_mask2]).reshape(-1,1)))
    else:
        points_1 = np.hstack((X[inlier_mask1], np.array(y[inlier_mask1]).reshape(-1,1)))
        points_2 = np.hstack((X_remaing[inlier_mask2], np.array(y_remaining[inlier_mask2]).reshape(-1,1)))
    return slope1, intercept1, slope2, intercept2 ,points_1, points_2


def fit_two_lines_ransac_improved(points, residual_threshold=10.0, min_angle_deg=15.0, 
                                 max_residual_score=0.5, min_inliers_ratio=0.1, 
                                 max_attempts=50, plot=True, verbose=True):
    """
    Fit two lines to point data using RANSACRegressor with LinearRegression.
    Validates angle separation and residual quality for robust fitting.
    
    Parameters:
    -----------
    points : array-like, shape (n_points, 2)
        x, y coordinates of points
    residual_threshold : float, default=10.0
        Maximum distance from line to be considered inlier
    min_angle_deg : float, default=15.0
        Minimum angle difference between lines in degrees
    max_residual_score : float, default=0.5
        Maximum allowed mean residual score (lower is better)
    min_inliers_ratio : float, default=0.3
        Minimum ratio of points that must be inliers for each line
    max_attempts : int, default=50
        Maximum number of random state attempts to find good fit
    plot : bool, default=True
        Whether to plot the results
    verbose : bool, default=True
        Whether to print fitting progress
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'line1_params': (slope, intercept) for line 1
        - 'line2_params': (slope, intercept) for line 2
        - 'inliers1': boolean mask for line 1 inliers
        - 'inliers2': boolean mask for line 2 inliers
        - 'fit_quality': dict with quality metrics
        - 'success': bool indicating if criteria were met
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    points = np.array(points)
    # radomly  sort the points
    np.random.shuffle(points)



    if len(points) < 6:  # Need at least 3 points per line
        if verbose:
            print("Error: Need at least 6 points to fit two lines")
        return None
    
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    min_angle_rad = np.deg2rad(min_angle_deg)
    min_inliers_per_line = int(len(points) * min_inliers_ratio)
    
    best_fit = None
    best_score = float('inf')
    
    if verbose:
        print(f"Attempting to fit two lines to {len(points)} points...")
        print(f"Criteria: min_angle≥{min_angle_deg}°, max_residual≤{max_residual_score}, min_inliers≥{min_inliers_per_line}")
    
    for attempt in range(max_attempts):
        try:
            # Fit first line using RANSAC
            ransac1 = RANSACRegressor(
                LinearRegression(), 
                residual_threshold=residual_threshold, 
                random_state=attempt,
                min_samples=2
            )
            ransac1.fit(X, y)
            inlier_mask1 = ransac1.inlier_mask_
            
            # Check if first line has enough inliers
            if np.sum(inlier_mask1) < min_inliers_per_line:
                continue
            
            # Fit second line to remaining points
            X_remaining = X[~inlier_mask1]
            y_remaining = y[~inlier_mask1]
            
            if len(X_remaining) < min_inliers_per_line:
                continue
                
            ransac2 = RANSACRegressor(
                LinearRegression(),
                residual_threshold=residual_threshold,
                random_state=attempt + 1,
                min_samples=2
            )
            ransac2.fit(X_remaining, y_remaining)
            inlier_mask2_subset = ransac2.inlier_mask_
            
            # Map second line inliers back to original indices
            inlier_mask2 = np.zeros(len(points), dtype=bool)
            remaining_indices = np.where(~inlier_mask1)[0]
            inlier_mask2[remaining_indices[inlier_mask2_subset]] = True
            
            # Check if second line has enough inliers
            if np.sum(inlier_mask2) < min_inliers_per_line:
                continue
            
            # Get line parameters
            line1_slope = ransac1.estimator_.coef_[0]
            line1_intercept = ransac1.estimator_.intercept_
            line2_slope = ransac2.estimator_.coef_[0]
            line2_intercept = ransac2.estimator_.intercept_
            
            # Calculate angle between lines
            angle1 = np.arctan(line1_slope)
            angle2 = np.arctan(line2_slope)
            angle_diff = abs(angle1 - angle2)
            
            # Handle angle wraparound
            if angle_diff > np.pi/2:
                angle_diff = np.pi - angle_diff
            
            # Check angle criterion
            if angle_diff < min_angle_rad:
                continue
            
            # Calculate residual quality for both lines
            y_pred1 = line1_slope * X.flatten() + line1_intercept
            y_pred2 = line2_slope * X.flatten() + line2_intercept
            
            # Calculate residuals for inliers only
            residuals1 = np.abs(y[inlier_mask1] - y_pred1[inlier_mask1])
            residuals2 = np.abs(y[inlier_mask2] - y_pred2[inlier_mask2])
            
            mean_residual1 = np.mean(residuals1)
            mean_residual2 = np.mean(residuals2)
            overall_residual = (mean_residual1 + mean_residual2) / 2
            
            # Check residual criterion
            if overall_residual > max_residual_score * residual_threshold:
                continue
            
            # Calculate fit quality score (lower is better)
            angle_score = 1.0 / (1.0 + angle_diff)  # Prefer larger angles
            residual_score = overall_residual / residual_threshold
            coverage_score = 1.0 - (np.sum(inlier_mask1) + np.sum(inlier_mask2)) / len(points)
            
            total_score = angle_score + residual_score + coverage_score
            
            if total_score < best_score:
                best_score = total_score
                best_fit = {
                    'line1_params': (line1_slope, line1_intercept),
                    'line2_params': (line2_slope, line2_intercept),
                    'inliers1': inlier_mask1.copy(),
                    'inliers2': inlier_mask2.copy(),
                    'fit_quality': {
                        'angle_diff_deg': np.rad2deg(angle_diff),
                        'mean_residual1': mean_residual1,
                        'mean_residual2': mean_residual2,
                        'overall_residual': overall_residual,
                        'inliers1_count': np.sum(inlier_mask1),
                        'inliers2_count': np.sum(inlier_mask2),
                        'outliers_count': len(points) - np.sum(inlier_mask1) - np.sum(inlier_mask2),
                        'coverage_ratio': (np.sum(inlier_mask1) + np.sum(inlier_mask2)) / len(points),
                        'quality_score': total_score
                    },
                    'success': True,
                    'ransac1': ransac1,
                    'ransac2': ransac2
                }
                
                if verbose and attempt % 10 == 0:
                    print(f"Attempt {attempt}: angle={np.rad2deg(angle_diff):.1f}°, "
                          f"residual={overall_residual:.2f}, score={total_score:.3f}")
        
        except Exception as e:
            if verbose and attempt % 20 == 0:
                print(f"Attempt {attempt} failed: {e}")
            continue
    
    if best_fit is None:
        if verbose:
            print("❌ Failed to find lines meeting all criteria")
        return {
            'line1_params': None,
            'line2_params': None,
            'inliers1': None,
            'inliers2': None,
            'fit_quality': None,
            'success': False
        }
    
    # Print results
    if verbose:
        print(f"\n✅ Successfully found two lines after {max_attempts} attempts:")
        print(f"Line 1: y = {best_fit['line1_params'][0]:.3f}x + {best_fit['line1_params'][1]:.3f}")
        print(f"Line 2: y = {best_fit['line2_params'][0]:.3f}x + {best_fit['line2_params'][1]:.3f}")
        print(f"Angle difference: {best_fit['fit_quality']['angle_diff_deg']:.1f}°")
        print(f"Mean residuals: {best_fit['fit_quality']['mean_residual1']:.2f}, {best_fit['fit_quality']['mean_residual2']:.2f}")
        print(f"Coverage: {best_fit['fit_quality']['coverage_ratio']:.1%}")
        print(f"Inliers: {best_fit['fit_quality']['inliers1_count']} + {best_fit['fit_quality']['inliers2_count']} = {best_fit['fit_quality']['inliers1_count'] + best_fit['fit_quality']['inliers2_count']}")
        print(f"Outliers: {best_fit['fit_quality']['outliers_count']}")
    
    # Plot results
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Plot all points
        plt.scatter(points[:, 0], points[:, 1], c='lightgray', alpha=0.6, s=30, label='All points')
        
        # Plot inliers for each line
        inliers1 = best_fit['inliers1']
        inliers2 = best_fit['inliers2']
        
        if np.any(inliers1):
            plt.scatter(points[inliers1, 0], points[inliers1, 1], 
                       c='red', s=50, alpha=0.8, label=f'Line 1 inliers ({np.sum(inliers1)})')
        
        if np.any(inliers2):
            plt.scatter(points[inliers2, 0], points[inliers2, 1], 
                       c='blue', s=50, alpha=0.8, label=f'Line 2 inliers ({np.sum(inliers2)})')
        
        # Plot outliers
        outliers = ~(inliers1 | inliers2)
        if np.any(outliers):
            plt.scatter(points[outliers, 0], points[outliers, 1], 
                       c='black', marker='x', s=60, label=f'Outliers ({np.sum(outliers)})')
        
        # Plot fitted lines
        x_range = np.linspace(points[:, 0].min() - 10, points[:, 0].max() + 10, 100)

    

        #line two ends at line 1. 
        # x_range_line2 = np.linespace
        
        line1_slope, line1_intercept = best_fit['line1_params']
        line2_slope, line2_intercept = best_fit['line2_params']
        
        y_line1 = line1_slope * x_range + line1_intercept
        y_line2 = line2_slope * x_range + line2_intercept
        
        angle1_deg = np.rad2deg(np.arctan(line1_slope))
        angle2_deg = np.rad2deg(np.arctan(line2_slope))
        
        plt.plot(x_range, y_line1, 'r--', linewidth=3, alpha=0.8,
                label=f'Line 1: y={line1_slope:.3f}x+{line1_intercept:.1f} (θ={angle1_deg:.1f}°)')
        plt.plot(x_range, y_line2, 'b--', linewidth=3, alpha=0.8,
                label=f'Line 2: y={line2_slope:.3f}x+{line2_intercept:.1f} (θ={angle2_deg:.1f}°)')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'RANSAC Two-Line Fitting (Angle diff: {best_fit["fit_quality"]["angle_diff_deg"]:.1f}°, '
                  f'Coverage: {best_fit["fit_quality"]["coverage_ratio"]:.1%})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.gca().invert_yaxis()
        xmin = points[:, 0].min() - 10
        xmax = points[:, 0].max() + 10
        ymin = points[:, 1].min() + 10
        ymax = points[:, 1].max() -10
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)

    return best_fit


def two_line_fit_with_rotation(points:np.ndarray, plot=False, verbose=False):
    """
    Fit two lines to the given 2D data by rotating the points and using RANSAC.

    Parameters:
    points (numpy.ndarray): A 2D array of shape (N, 2) where each row represents a point (x, y).
    plot (bool): If True, plot the data points and the fitted lines.
    verbose (bool): If True, print additional information during processing.

    Returns:
    tuple: Coefficients of the two lines and points for the two lines in the form (slope1, intercept1, slope2, intercept2, points1, points2).
    """
        
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import RANSACRegressor, LinearRegression

 
    points = np.array(points)
    # radomly  sort the points
    np.random.shuffle(points)
    
    X = points[:,0].reshape(-1,1)
    y = points[:,1]
    #first guess
    # Use RANSAC to detect first line
    ransac1 = RANSACRegressor(LinearRegression(), residual_threshold=10, random_state=0)
    ransac1.fit(X, y)
    inlier_mask1 = ransac1.inlier_mask_
    outlier_mask1= np.logical_not(inlier_mask1)

    X_remaining = X[outlier_mask1]
    y_remaining = y[outlier_mask1]
    # Use RANSAC on remaining points for second line
    ransac2 = RANSACRegressor(LinearRegression(), residual_threshold=5, random_state=1)
    ransac2.fit(X_remaining, y_remaining)
    inlier_mask2 = ransac2.inlier_mask_
    outlier_mask2= np.logical_not(inlier_mask2)


    #Full mask for original points
    full_inlier_mask2 = np.zeros(len(points), dtype=bool)
    remaining_indices = np.where(outlier_mask1)[0]
    full_inlier_mask2[remaining_indices[inlier_mask2]] = True
    full_outlier_mask2 = np.logical_not(full_inlier_mask2)


    inlier_mask2 = full_inlier_mask2
    inlier_mask2_temp = inlier_mask2.copy()
    outlier_mask2 = full_outlier_mask2
# check spacing between adjacent points
    spacing1 = np.diff(np.sort(X[inlier_mask1].flatten()))
    mean_spacing1 = np.mean(spacing1)
    spacing2 = np.diff(np.sort(X[inlier_mask2].flatten()))
    mean_spacing2 = np.mean(spacing2)
  
    if any(spacing2 > 100):
        if verbose:
            print("Large gaps detected in line 2, refining fit...")
        large_gaps_indices = np.where(spacing2 > 100)[0]
        space_count = len(spacing2)
        need_to_drop=[]
        for index in range(large_gaps_indices[0], space_count):
            need_to_drop.append(index+1)
        
        Dropped_x_values = np.sort(X[inlier_mask2].flatten())[need_to_drop]
        dropped_locs = []
        for val in Dropped_x_values:
            dropped_locs.append(np.where(X.flatten() == val))
        #flatten the list of tuples if they are the same size
        try:
            dropped_locs = np.array(dropped_locs).flatten()
        except:
            temp_locs = dropped_locs.copy()
            dropped_locs = []
            for array in temp_locs:
                for val in array[0]:
                    dropped_locs.append(val)
        #create new masks without the dropped points
        new_inlier_mask2 = inlier_mask2.copy()
        new_inlier_mask2[dropped_locs] = False
        new_outlier_mask2 = np.logical_not(new_inlier_mask2)    
        Dropped_x_values = np.sort(X[inlier_mask2].flatten())

        if verbose:
            print(f"Dropped {len(dropped_locs)} points from line 2 fit due to large gaps")
            print(f"New line 2 inliers count: {np.sum(new_inlier_mask2)}")

        logger.debug(f"Dropped x values: {Dropped_x_values}")
        logger.debug(f"Dropped {len(dropped_locs)} points from line 2 fit due to large gaps")
        logger.debug(f"New line 2 inliers count: {np.sum(new_inlier_mask2)}")


        inlier_mask2 = new_inlier_mask2
        outlier_mask2 = new_outlier_mask2
    #slopes and intercepts
    slope1 = ransac1.estimator_.coef_[0]
    intercept1 = ransac1.estimator_.intercept_
    slope2 = ransac2.estimator_.coef_[0]
    intercept2 = ransac2.estimator_.intercept_
    if verbose:
        print(f"Line 1: y = {slope1:.3f}x + {intercept1:.1f}")
        print(f"Line 2: y = {slope2:.3f}x + {intercept2:.1f}")
    logger.info(f"Line 1: y = {slope1:.3f}x + {intercept1:.1f}")
    logger.info(f"Line 2: y = {slope2:.3f}x + {intercept2:.1f}")

    if np.abs(slope1 - slope2) < 0.3:
        if verbose: print("Warning: Lines are nearly parallel, fit may be unreliable")
        logger.warning("Lines are nearly parallel, fit may be unreliable")

    if np.abs(np.abs(slope2)) > 24.9 or np.abs(np.abs(slope1) -1.45) < 0.05:
        if verbose: print("Warning: Line 2 slope is near vertical, results may be unreliable")
        logger.warning("Line 2 slope is near vertical, results may be unreliable")

        if plot:
        #     #plot the first inliners and its prediction
            fig, ax  = plt.subplots()
            ax.scatter(X, y, c="gray", s=10, alpha= 0.5,label="data points")
            ax.scatter(X[inlier_mask1], y[inlier_mask1],
                    marker="d",c="r", s=20, label="data_line1",)
            ax.scatter(X[inlier_mask2], y[inlier_mask2],
                    marker="o",c="b", s=20, label="data_line2")
            
            ax.set_title(f"Fit before rotation")
            ax.yaxis.set_inverted(True)
            ax.grid()
            ax.legend()

        #peform rotation on points and re-fit
        angle_to_rotate = np.deg2rad(45)
        # Additional code for rotation and re-fitting would go here
        rotated_X = X * np.cos(angle_to_rotate) - y.reshape(-1,1) * np.sin(angle_to_rotate)
        rotated_y = X * np.sin(angle_to_rotate) + y.reshape(-1,1) * np.cos(angle_to_rotate)
        rot_ransac1 = RANSACRegressor(LinearRegression(), residual_threshold=10, random_state=0)
        rot_ransac1.fit(rotated_X, rotated_y)
        rot_inlier_mask1 = rot_ransac1.inlier_mask_
        rot_outlier_mask1= np.logical_not(rot_inlier_mask1)

        rotated_X_remaining = rotated_X[rot_outlier_mask1]
        rotated_y_remaining = rotated_y[rot_outlier_mask1]
        # Use RANSAC on remaining points for second line
        rot_ransac2 = RANSACRegressor(LinearRegression(), residual_threshold=10, random_state=1)
        rot_ransac2.fit(rotated_X_remaining, rotated_y_remaining)
        rot_inlier_mask2 = rot_ransac2.inlier_mask_
        rot_outlier_mask2= np.logical_not(rot_inlier_mask2)
        #Full mask for original points
        full_rot_inlier_mask2 = np.zeros(len(points), dtype=bool)
        remaining_indices = np.where(rot_outlier_mask1)[0]
        full_rot_inlier_mask2[remaining_indices[rot_inlier_mask2]] = True
        full_rot_outlier_mask2 = np.logical_not(full_rot_inlier_mask2)

        rotatedX_1 = np.arange(rotated_X[rot_inlier_mask1].min(), rotated_X[rot_inlier_mask1].max(), 0.1)
        rotatedY_1 = rot_ransac1.estimator_.coef_[0] * rotatedX_1 + rot_ransac1.estimator_.intercept_[0]
        rotatedX_2 = np.arange(rotated_X_remaining[rot_inlier_mask2].min(), rotated_X_remaining[rot_inlier_mask2].max(), 0.1)
        rotatedY_2 = rot_ransac2.estimator_.coef_[0] * rotatedX_2 + rot_ransac2.estimator_.intercept_[0]


        if plot:
        #     #plot the first inliners and its prediction
            fig, ax  = plt.subplots()
            ax.scatter(rotated_X, rotated_y, c="gray", s=10, alpha= 0.5,label="data points")
            ax.scatter(rotated_X[rot_inlier_mask1], rotated_y[rot_inlier_mask1],
                    marker="d",c="r", s=20, label="data_line1",)
            ax.scatter(rotated_X_remaining[rot_inlier_mask2], rotated_y_remaining[rot_inlier_mask2],
                    marker="o",c="b", s=20, label="data_line2")
            
            ax.plot(rotatedX_1, rotatedY_1, "g-", label="Line 1 fit")
            ax.plot(rotatedX_2, rotatedY_2, "m-", label="Line 2 fit")
            ax.set_title(f"Rotated fit")
            ax.yaxis.set_inverted(True)
            ax.grid()
            ax.legend()
        #slopes and intercepts
        rot_slope1 = rot_ransac1.estimator_.coef_[0]
        rot_intercept1 = rot_ransac1.estimator_.intercept_
        rot_slope2 = rot_ransac2.estimator_.coef_[0]
        rot_intercept2 = rot_ransac2.estimator_.intercept_

        #rotate lines back
        slope1 = (rot_slope1[0]* np.cos(angle_to_rotate) - np.sin(angle_to_rotate)) / (np.cos(angle_to_rotate) + rot_slope1[0] * np.sin(angle_to_rotate))
        intercept1 = rot_intercept1[0] /( np.cos(angle_to_rotate) + rot_slope1[0] * np.sin(angle_to_rotate))

        slope2 = (rot_slope2[0]* np.cos(angle_to_rotate) - np.sin(angle_to_rotate)) / (np.cos(angle_to_rotate) + rot_slope2[0] * np.sin(angle_to_rotate))
        intercept2 = rot_intercept2[0] /( np.cos(angle_to_rotate) + rot_slope2[0] * np.sin(angle_to_rotate))

        if verbose:
            print(f"Line 1: y = {rot_slope1[0]:.3f}x + {rot_intercept1[0]:.1f}")
            print(f"Line 2: y = {rot_slope2[0]:.3f}x + {rot_intercept2[0]:.1f}")
            print(f"After rotation correction:")
            print(f"Line 1: y = {slope1:.3f}x + {intercept1:.1f}")
            print(f"Line 2: y = {slope2:.3f}x + {intercept2:.1f}")



        # IF RE-ROTATION IMPROVES THE FIT, USE IT WIP for checking this
        # Calculate residuals for original fit
        y_pred1 = slope1 * X[rot_inlier_mask1].flatten() + intercept1
        y_pred2 = slope2 * X[full_rot_inlier_mask2].flatten() + intercept2 

        residuals1 = np.abs(y[rot_inlier_mask1] - y_pred1)
        residuals2 = np.abs(y[full_rot_inlier_mask2] - y_pred2)
        mean_residual1 = np.mean(residuals1)
        mean_residual2 = np.mean(residuals2)   
        overall_residual = (mean_residual1 + mean_residual2) / 2

        inlier_mask1 = rot_inlier_mask1
        inlier_mask2 = full_rot_inlier_mask2
        line1_x = np.arange(X[rot_inlier_mask1].min(),X[rot_inlier_mask1].max(),0.1)    
        line1_y = slope1 * line1_x + intercept1
        line2_x = np.arange(X[inlier_mask2].min(),X[inlier_mask2].max(),0.1)
        line2_y = slope2 * line2_x + intercept2

        if np.abs(slope2) > 24.9:
            line2_y = np.arange(y[full_rot_inlier_mask2].min(),y[full_rot_inlier_mask2].max(),0.1)  
            line2_x = (line2_y - intercept2) / slope2

        if plot:
            fig, ax  = plt.subplots()
            ax.scatter(X,y, c="gray", s=10, alpha= 0.5,label="data points")
            ax.scatter(X[rot_inlier_mask1], y[rot_inlier_mask1],
                    marker="d",c="r", s=20, label="data_line1",)
            ax.scatter(X[full_rot_inlier_mask2], y[full_rot_inlier_mask2],
                    marker="o",c="b", s=20, label="data_line2")

            ax.plot(line1_x, line1_y, "g-", label="Line 1 fit")
            ax.plot(line2_x, line2_y, "b-", label="Line 2 fit")

            ax.set_title(f"Re-rotated fit")
            ax.yaxis.set_inverted(True)
            ax.grid()
            ax.legend()
    ####End of rotation correction

    #calc intersection of the two lines
    intersect_x = (intercept2 - intercept1) / (slope1 - slope2)
    intersect_y = slope1 * intersect_x + intercept1

    line_x2 = np.arange(X[inlier_mask2].min(),X[inlier_mask2].max(),0.01)
    line_y2 = slope2 * line_x2 + intercept2

    line_x1 = np.arange(X[inlier_mask1].min(),X[inlier_mask1].max(),0.1)
    line_y1 = slope1 * line_x1 + intercept1

    #if line 2 does not intersect with line 1, extend it to the intersection
    if len(line_x2) == 0:
        logger.error("No inliers found for line 2")
        inlier_mask2 = inlier_mask2_temp
        line_x2 = np.array(np.arange(X[inlier_mask2].min(),X[inlier_mask2].max(),0.01))
        line_y2 = slope2 * line_x2 + intercept2

    if min(line_x2) > intersect_x:
        line_x2 = np.insert(line_x2, 0, intersect_x)
        line_y2 = slope2 * line_x2 + intercept2
    elif max(line_x2) < intersect_x:
        line_x2 = np.append(line_x2, intersect_x)
        line_y2 = slope2 * line_x2 + intercept2

    if np.abs(slope2) > 24.9:
        
        
        line_y2 = np.arange(y[full_rot_inlier_mask2].min(),y[full_rot_inlier_mask2].max(),0.1)  
        line_x2 = (line_y2 - intercept2) / slope2
        if min(line_y2) > intersect_y:
        
            line_y2 = np.insert(line_y2, 0, intersect_y)
            line_x2 = (line_y2 - intercept2) / slope2
        elif max(line_y2) < intersect_y:
        
            line_y2 = np.append(line_y2, intersect_y)
            line_x2 = (line_y2 - intercept2) / slope2
        



    if plot:
        #plot the first inliners and its prediction
        fig, ax  = plt.subplots()
        ax.scatter(X,y, c="gray", s=10, alpha= 0.5,label="data points")
        ax.scatter(X[inlier_mask1], y[inlier_mask1],
                    marker="d",c="r", s=20, label="data_line1",)


        ax.scatter(X[inlier_mask2], y[inlier_mask2],
                    marker="o",c="b", s=20, label="data_line2")
        
        ax.plot(line_x1, line_y1, "g-", label="Line 1 fit")
        ax.plot(line_x2, line_y2, "b-", label="Line 2 fit")
        ax.set_title(f"Final fit")
        ax.yaxis.set_inverted(True)
        ax.grid()
        ax.legend()
  

    #length of lines
    line1_length = np.sqrt((line_x1[-1]-line_x1[0])**2 + (line_y1[-1]-line_y1[0])**2)
    line2_length = np.sqrt((line_x2[-1]-line_x2[0])**2 + (line_y2[-1]-line_y2[0])**2)
    if verbose:
        print("Number of inliers for line 1:", np.sum(inlier_mask1), "out of", len(y))
        print("Number of inliers for line 2:", np.sum(inlier_mask2), "out of", len(X[outlier_mask1]))

    # logger.info("Number of inliers for line 1:", np.sum(inlier_mask1), "out of", len(y))
    # logger.info("Number of inliers for line 2:", np.sum(inlier_mask2), "out of", len(X[outlier_mask1]))
    logger.info(f"After rotation correction:")
    logger.info(f"Line 1: y = {slope1:.3f}  x + {intercept1:.1f}")
    logger.info(f"Line 2: y = {slope2:.3f}  x + {intercept2:.1f}") 

    points_1 = np.hstack((X[inlier_mask1], np.array(y[inlier_mask1]).reshape(-1,1)))
    points_2 = np.hstack((X[inlier_mask2], np.array(y[inlier_mask2]).reshape(-1,1)))
    return slope1, intercept1, slope2, intercept2 ,points_1, points_2





def Angle_Measurment(image,points, line_info, plot=False, verbose=False, image_number=""):
    """
    Measure the angle between two lines defined by their slope and intercept.

    Parameters:
    points (numpy.ndarray): A 2D array of shape (N, 2) where each row represents a point (x, y).
    line_info (tuple): A tuple containing the slopes and intercepts of the two lines in the form (slope1, intercept1, slope2, intercept2).
    plot (bool): If True, plot the data points and the fitted lines.
    verbose (bool): If True, print additional information during processing.

    Returns:
    float: The angle between the two lines in degrees.
    """
        
    import numpy as np
    import matplotlib.pyplot as plt

    if type(image) is str:
        image = cv2.imread(image)

    if points is None:
        Verticle_scan = vertical_scan_for_center_peaks(image, resolution=5, plot=0, verbose=False)
        Horizontal_scan = horizontal_scan_for_center_peaks(image, resolution=5, plot=0, verbose=False)    
        #stack the two scans to get the center
        points = np.vstack((Horizontal_scan,Verticle_scan))
    
    if line_info is None:
        line_info = two_line_fit_with_rotation(points, plot=False, verbose=False)

    
    line1_points = line_info[4]
    line2_points = line_info[5]
    line2_y_mean = np.mean(line2_points[:,1])
    line2_x_mean = np.mean(line2_points[:,0])

    #Find the angle of the main line, using the secondary line to determine orientation
    #Make the anlge be from the postive x axis, counter clockwise positive
    Angle_line1 = np.rad2deg(np.arctan(line_info[0]))
    Angle_line2 = np.rad2deg(np.arctan(line_info[2]))
    line1_m = line_info[0]
    line1_b = line_info[1]
    line2_m = line_info[2]
    line2_b = line_info[3]
    x_intersect = (line2_b - line1_b) / (line1_m - line2_m)
    y_intersect = line1_m * x_intersect + line1_b
    if verbose:
        print(f"Intersection at x={x_intersect:.1f}, y={y_intersect:.1f}")
        print(f"Line 1 slope: {line1_m:.3f}, intercept: {line1_b:.1f}")
        print(f"Line 2 slope: {line2_m:.3f}, intercept: {line2_b:.1f}")
        print(f"Angle of line 1 before quadrant adjustment: {Angle_line1:.1f}°")
        print(f"Angle of line 2 before quadrant adjustment: {Angle_line2:.1f}°")

    logger.info(f"Intersection at x={x_intersect:.1f}, y={y_intersect:.1f}")
    logger.info(f"Line 1 slope: {line1_m:.3f}, intercept: {line1_b:.1f}")
    logger.info(f"Line 2 slope: {line2_m:.3f}, intercept: {line2_b:.1f}")
    logger.info(f"Angle of line 1 before quadrant adjustment: {Angle_line1:.1f} degrees")
    logger.info(f"Angle of line 2 before quadrant adjustment: {Angle_line2:.1f} degrees")

    # Adjust the angle based on which quadrant line 2 lies in
    line2_quadrant = None
    if line2_x_mean > x_intersect and line2_y_mean < y_intersect:
        line2_quadrant = 1
 
    elif line2_x_mean < x_intersect and line2_y_mean < y_intersect:
        line2_quadrant = 2

    elif line2_x_mean < x_intersect and line2_y_mean > y_intersect:
        line2_quadrant = 3

    elif line2_x_mean > x_intersect and line2_y_mean > y_intersect:
        line2_quadrant = 4
    else:
        line2_quadrant = "On axis"

    #logic to rotate angle based on quad and slope of line1 and line2, due to limitations of arctan
    if np.abs(line1_m) > 100:
        #near vert
        if line2_quadrant == 2:
            if (np.abs(line2_m) -1.45) >0:
                Angle_line1 = Angle_line1
            else:
                Angle_line1 = 180-Angle_line1
        elif line2_quadrant==4:
            if (np.abs(line2_m) -1.4) >0:
                Angle_line1 = 270+Angle_line1
            else:
                Angle_line1 = 360-Angle_line1
    elif line1_m <= 0:
        if line2_quadrant == 3 or line2_quadrant ==4:
            Angle_line1 = 180-Angle_line1
        if line2_quadrant ==1 or line2_quadrant==2:
            Angle_line1=np.abs(Angle_line1)
    elif line1_m >0:
        if line2_quadrant == 2 or line2_quadrant==3:
            Angle_line1 = 180-Angle_line1
        elif line2_quadrant ==4 or line2_quadrant ==1:
            Angle_line1 = 360- Angle_line1 

    #count of points of line 1 in each quadrant
    line1_quadrant_counts = [0,0,0,0]
    for point in line1_points:
        if point[0] > x_intersect and point[1] < y_intersect:
            line1_quadrant_counts[0] += 1
        elif point[0] < x_intersect and point[1] < y_intersect:
            line1_quadrant_counts[1] += 1
        elif point[0] < x_intersect and point[1] > y_intersect:
            line1_quadrant_counts[2] += 1
        elif point[0] > x_intersect and point[1] > y_intersect:
            line1_quadrant_counts[3] += 1

    #count of points of line 2 in each quadrant
    line2_quadrant_counts = [0,0,0,0]
    for point in line2_points:
        if point[0] > x_intersect and point[1] < y_intersect:
            line2_quadrant_counts[0] += 1
        elif point[0] < x_intersect and point[1] < y_intersect:
            line2_quadrant_counts[1] += 1
        elif point[0] < x_intersect and point[1] > y_intersect:
            line2_quadrant_counts[2] += 1
        elif point[0] > x_intersect and point[1] > y_intersect:
            line2_quadrant_counts[3] += 1


    #Need to add in some uncertainty estimation here
    #Based on the number of points in each quadrant, and the length of the lines

    if verbose:
        print("--------------------------------------------------")
        print(f"Total points: {len(points)}")
        print(f"Line 1 total points: {len(line1_points)}")
        print(f"Line 1 quadrant counts: Q1={line1_quadrant_counts[0]}, Q2={line1_quadrant_counts[1]}, Q3={line1_quadrant_counts[2]}, Q4={line1_quadrant_counts[3]}")
        print(f"Line 2 quadrant counts: Q1={line2_quadrant_counts[0]}, Q2={line2_quadrant_counts[1]}, Q3={line2_quadrant_counts[2]}, Q4={line2_quadrant_counts[3]}")
        print(f"Line 2 lies in quadrant: {line2_quadrant}")
        print(f"Line 1 angle = {Angle_line1:.2f} degrees, Line 2 angle = {Angle_line2:.2f} degrees")
        print(f"Intersection point: ({x_intersect:.2f}, {y_intersect:.2f})")

    logger.info("--------------------------------------------------")
    logger.info(f"Total points: {len(points)}")
    logger.info(f"Line 1 total points: {len(line1_points)}")
    logger.info(f"Line 1 quadrant counts: Q1={line1_quadrant_counts[0]}, Q2={line1_quadrant_counts[1]}, Q3={line1_quadrant_counts[2]}, Q4={line1_quadrant_counts[3]}")
    logger.info(f"Line 2 quadrant counts: Q1={line2_quadrant_counts[0]}, Q2={line2_quadrant_counts[1]}, Q3={line2_quadrant_counts[2]}, Q4={line2_quadrant_counts[3]}")
    logger.info(f"Line 2 lies in quadrant: {line2_quadrant}")
    logger.info(f"Line 1 angle = {Angle_line1:.2f} degrees, Line 2 angle = {Angle_line2:.2f} degrees")
    logger.info(f"Intersection point: ({x_intersect:.2f}, {y_intersect:.2f})")
    logger.info("--------------------------------------------------")
    

    if plot:
        fig, ax = plt.subplots()
        if image is not None:
            ax.imshow(image, cmap='gray')

        ax.scatter(points[:,0], points[:,1], color='gray', s=10, label='Data Points')
        X_line1 = np.arange(line1_points[:,0].min(), line1_points[:,0].max(), 0.1)
        X_line2 = np.arange(line2_points[:,0].min(), line2_points[:,0].max(), 0.1)

        ax.plot(X_line1, line1_m * X_line1 + line1_b, 'r-', label=f'Line 1 Fit angle={Angle_line1:.1f}°')
        ax.plot(X_line2, line2_m * X_line2 + line2_b, 'b-', label=f'Line 2 Fit angle={Angle_line2:.1f}°')
        ax.plot(x_intersect, y_intersect, 'go', markersize=10, label='Intersection Point')
        ax.axhline(y=y_intersect, color='gray', linestyle='--', label='Horizontal Line at Intersection')
        ax.axvline(x=x_intersect, color='slategray', linestyle='--', label='Vertical Line at Intersection')
        ax.set_xlim(points[:,0].min()-10, points[:,0].max()+10)
        ax.set_ylim(points[:,1].min()-10, points[:,1].max()+10)
        if image is not None:
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)

        if image_number != "":
            ax.set_title(f"Angle Measurement {image_number}")
        else:
            ax.invert_yaxis()
        ax.grid()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()


    return Angle_line1, Angle_line2, (x_intersect, y_intersect)



def Analyze_Image(image_file_name: str, plot_level:int=0, verbose_level:int = 1)-> tuple:
    """
    Analyze an image to return the angle of the main line with the horizontal axis. 
    The angle is measured in degrees, with 0 degrees being horizontal and positive angles measured counter-clockwise.
    Parameters:
    image_file_name (str): Path to the image file. 


    """


    import time
    plot = False
    verbose = False
    
    start = time.time()
    if not os.path.isfile(image_file_name):
        print(f"Error: File {image_file_name} does not exist.")
        return None
        

    image_number = os.path.splitext(os.path.basename(image_file_name))[0]
    img_number = image_number[-5:]  # Extract the last 5 characters for img_number
    image = cv2.imread(image_file_name)
    if plot_level > 2:
        plot = plot_level - 2
    if verbose_level > 2:
        verbose = True

    Verticle_scan = vertical_scan_for_center_peaks(image, resolution=20, plot=plot, verbose=verbose)
    Horizontal_scan = horizontal_scan_for_center_peaks(image, resolution=20, plot=plot, verbose=verbose)

    #stack the two scans to get the center
    Scan = np.vstack((Horizontal_scan,Verticle_scan))

    if plot_level > 1:
        plot = True
    if verbose_level > 1:
        verbose = True
    line_info = two_line_fit_with_rotation(Scan, plot=plot, verbose=verbose)

    if plot_level > 0:
        plot = True
    if verbose_level > 0:
        verbose = True

    Angle_info=Angle_Measurment(image,Scan,line_info,plot=plot,verbose=verbose,image_number=img_number)

    finish = time.time()
    if verbose:
        print(f"Time to process image {img_number}: {finish - start:.2f} seconds")
        print("Angle info:", Angle_info)
    logging.info(f"Processed image {img_number} in {finish - start:.2f} seconds")
    logging.info(f"Angle info: {Angle_info}")

    return Angle_info




def Analyze_Image_Simple(image_name):
    """
    A simplified version of Analyze_Image with default parameters for quick analysis.

    Parameters:
    image_name (str): Path to the image file.

    Returns:
    tuple: The angle of the main line with the horizontal axis and intersection point.
    """
    ## setup logging




    angle_info =Analyze_Image(image_name, plot_level=1, verbose_level=0)

    angle = angle_info[0]
    

    return np.round(angle,2)




