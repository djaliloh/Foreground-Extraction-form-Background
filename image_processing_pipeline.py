import argparse
import glob
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils import*
import tqdm
 
def histo_calculation(params):
    """
    Calculate histograms for images in the specified directory.
    
    """

    path  = params['dataPath']
    depth = params['params']['depth'] 
    phenology_pth = params['phenologyDir']

    entire_dir_hist = []
    
    for img_path in tqdm.tqdm(glob.glob(join_paths(path, phenology_pth, "*"))):
        if depth is None or depth is False: 
            img = cv.imread(img_path)
            if img is None:
                continue
            
            for channel in range(3):
                img_hist = cv.calcHist([img], [channel], None, [256], [0, 256])
                img_hist /= img_hist.sum()  # Normalize histogram
                entire_dir_hist.append(img_hist)
        
        elif depth is True:
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None:
                continue

            # if img_gray.ndim == 3:
            #     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # print('ICI->',img_gray.ndim)
            img_hist = cv.calcHist([img], [0], None, [256], [0, 256])
            img_hist /= img_hist.sum()  # Normalize histogram
            entire_dir_hist.append(img_hist)
    
    return entire_dir_hist
    
def thresholding(params, thr_direct=None, thr_window=None):
    
    thr_value     = params['manualThrValue'] 
    normalization = params['params']['normalization']
    mean_value    = params['params']['meanValue']
    phenology_pth = params['phenologyDir']
    path2depthmap = params['dataPath']
    thr_method    = params['thresholdMethod']
    available_thr = params['availableThresholds']
    
    start_time = time.time()
    for idx, img_path in enumerate(glob.glob(join_paths(path2depthmap, phenology_pth, "*"))):
        depth_map = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        if depth_map is None:
            raise ValueError(f"Error loading image: {img_path}")
        
        if depth_map.ndim == 3:
            depth_map = cv.cvtColor(depth_map, cv.COLOR_BGR2GRAY)

        # Normalization
        if normalization:
            depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        # mean value
        thr_value = np.mean(depth_map) if mean_value else params['manualThrValue']
        if mean_value:
            print(f"Mean value threshold: {thr_value}")
        # print(f"Mean value threshold: {thr_value}" if mean_value else f"Manual threshold value: {params['manualThrValue']}")

        selected_thr = None
        thresh_type = None

        if thr_method == 'global' and 'global' in available_thr:
            selected_thr = thr_value if not mean_value and not thr_window else (thr_window if params['useThrWindow'] else thr_direct)
            _, thresh_type = cv.threshold(depth_map, selected_thr, 255, cv.THRESH_BINARY)
            # print("select ->", selected_thr)
            # print('window', params['useThrWindow'])
        
        elif thr_method == 'adaptive-mean' and 'adaptive-mean' in available_thr:
            thresh_type = cv.adaptiveThreshold(depth_map, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                cv.THRESH_BINARY, params['blockSize'], params['C'])

        elif thr_method == 'adaptive-gaussian' and 'adaptive-gaussian' in available_thr:
            thresh_type = cv.adaptiveThreshold(depth_map, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY, params['blockSize'], params['C'])

        elif thr_method == 'otsu' and 'otsu' in available_thr:
            _, thresh_type = cv.threshold(depth_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        else:
            raise ValueError(f"Threshold method '{thr_method}' not recognized or not available.")


        # save mask in directory
        image_name = os.path.basename(img_path).split('.')[0]
        maskdir = make_dirs(join_paths(go_up_levels(path2depthmap, 2), f"mask_{thr_method}", phenology_pth))
        mask_name = f"{image_name}_{selected_thr}_{thr_method}_.png" if selected_thr is not None else f"{image_name}_{thr_method}_.png"
        cv.imwrite(join_paths(maskdir, mask_name), thresh_type)
        print(f"mask_{idx+1} saved !")
    print("--- %s seconds ---" % (time.time() - start_time))


def main():
    parser = argparse.ArgumentParser(description='image_processing_pipeline', epilog="image_conversion_histogram_threshold")
    parser.add_argument('--config', default="./config.yaml", help='Config path')
    parser.add_argument('--conversion', action='store_true', help='')
    parser.add_argument('--plot', action='store_true', help='')
    args = parser.parse_args()  

    parameters = read_yaml(args.config)
    all_hist = histo_calculation(parameters)
    window_size = parameters['params']['windowSize']

    #--Identify an optimal threshold in a combined and flattened histogram--#
    flattened_histograms = flatten_hist(all_hist) # standardize the data before calculating the average histogram
    avg_histogram = np.mean(flattened_histograms, axis=0) # Average histogram to get an overall average histogram


    # Linear regression (excluding intensity 0)
    X = np.arange(1, 256).reshape(-1, 1)  # Intensities from 1 to 255
    Y = avg_histogram[1:] 
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)

    residuals = Y - Y_pred 

    # Option 1: Find maximum residual directly (excluding intensity 0)
    threshold_direct = np.argmax(residuals) + 1  # Add 1 to account for index shift

    # Option 2: Find the threshold using the max sum of residuals over a sliding window
    threshold_window = max(range(1, 256 - window_size),
                        key=lambda i: np.sum(residuals[i-1:i + window_size-1]))

    print("Calculated Threshold (Direct):", threshold_direct)
    print("Calculated Threshold (Window Sum):", threshold_window)

    # perform thresholding
    thresholding(parameters, thr_direct=threshold_direct, thr_window=threshold_window)

    # Plotting
    if args.plot:
        plt.plot(np.arange(1, 256), avg_histogram[1:])  # Start from intensity 1
        plt.title("Average Histogram (Excluding Intensity 0)")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.show()

if __name__ == '__main__':
    main()


# b, g, r = cv.split(img)
# print(f"Channels are the same ? : {np.allclose(b, g) and np.allclose(g, r)} | Shape is:{img_gray.shape}")