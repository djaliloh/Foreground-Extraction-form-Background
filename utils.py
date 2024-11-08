import os
import numpy as np
import cv2 as cv
import yaml

def go_up_levels(path, levels):
    for _ in range(levels):
        path = os.path.dirname(path)
    return path

def join_paths(*paths):
    return os.path.join(*paths)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# Open and read the yaml file 
def read_yaml(cfg_file): 
    """
    Reads a YAML file and returns the configuration as a dictionary.

    Args:
    - cfg_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration parameters.
    """
    with open(cfg_file, 'r') as yaml_in:
        cfg = yaml.safe_load(yaml_in)
        return cfg


def flatten_hist(matrix):
    return [hist.flatten() for hist in matrix]


# def rawImage2otherExtension(path, extension):
#     for idx, raw_img in enumerate(glob.glob(os.path.join(cr3_file, "*"))):
#         print(f"image N{idx}: {os.path.basename(raw_img)}")
#         image_name  = os.path.basename(raw_img).split(".")[0]
#         with rawpy.imread(raw_img) as raw:
#             rgb_image = raw.postprocess()
#             img = Image.fromarray(rgb_image)
#             img.save(os.path.join(out_pth, f"{idx}_{image_name}.tiff"), format='TIFF')

def save_histo_image(pth, im_hist, is_depth, pheno_path, histdir="histogram"):
    pths = go_up_levels(pth, 2)
    histo_pth = join_paths(pths, histdir, pheno_path)
    make_dirs(histo_pth)

    if is_depth == 2:
        hist_image = np.zeros((300, 256), dtype=np.uint8)
    else:
        hist_image = np.zeros((300, 256, 3), dtype=np.uint8)

    cv.normalize(im_hist, im_hist, alpha=0, beta=300, norm_type=cv.NORM_MINMAX)  

    for x in range(256):
        if is_depth == 2:
            cv.line(hist_image, (x, 300), (x, 300 - int(im_hist[x].item())), 255, 1)
        else:
            cv.line(hist_image, (x, 300), (x, 300 - int(im_hist[x].item())), (255, 0, 0), 1)

    filename = join_paths(histo_pth, f'histogram_depth.png' if is_depth == 2 else f'histogram_channel_{is_depth}.png')
    cv.imwrite(filename, hist_image)