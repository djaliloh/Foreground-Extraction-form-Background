import argparse
import glob
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
            
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
            img_hist /= img_hist.sum()  # Normalize histogram
            entire_dir_hist.append(img_hist)
    
    return entire_dir_hist


def thresholding(params, thr_direct=None, thr_window=None): 

    whichThr      = params['threshold']['thresholdMode']
    isdirect      = params['threshold']['thresholdMode']['direct']
    thr_value     = params['threshold']['manualThrValue'] 
    thrChoice     = params['threshold']['thrChoice']
    normalization = params['params']['normalization']
    mean_value    = params['params']['meanValue']
    phenology_pth = params['phenologyDir']
    path2depthmap = params['dataPath']
    
    # print(thr_value, '->', whichThr["autoThreshold"])
    # print('->',whichThr)
    # exit(0)
    for idx, img in enumerate(glob.glob(join_paths(path2depthmap, phenology_pth, "*"))):
        depth_map = cv.imread(img, cv.IMREAD_UNCHANGED)
        h, w, _ = depth_map.shape
        image_name = os.path.basename(img).split('.')[0]

        if depth_map is None:
            raise ValueError(f"Error loading image: {img}")
        
        if normalization:
            depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
        
        # Average of the depth values
        if mean_value:  
            thr_value = np.mean(depth_map)
            print(f"Mean value threshold: {thr_value}")

        # Using automatic manual or threshold
        if thrChoice == "autoThreshold" and whichThr["autoThreshold"]:
            thr_value = thr_direct if isdirect else thr_window
            
        elif thrChoice == "manualThreshold" and whichThr["manualThreshold"]:    
            thr_value = thr_value

        # Seuillage
        _, high_intensity_pixels = cv.threshold(depth_map, thr_value, 255, cv.THRESH_BINARY)

        # save in directory
        maskdir = make_dirs(join_paths(go_up_levels(path2depthmap, 2), "mask_", phenology_pth))
        mask_name = f"{image_name}_{thr_value}_high_intensity_pixels.png"
        cv.imwrite(os.path.join(maskdir, mask_name), high_intensity_pixels)
        print(f"mask_{idx+1} saved !")


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