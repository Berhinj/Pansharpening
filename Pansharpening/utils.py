#! /usr/bin/env python
"""Utils Functions.

This script groups all basic custom utilities functions that will be used in other scripts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist(img, drop = None):
    """Plot an the histogram of an image pixel values.

    Arguments:
    -----------
        img: np.ndarray
            Image that will be used to derive the histogram
        drop: float
            Any value that shouldn't be considered in the histogram. Usefull for 0s or NAs for example.
    """
    img_ravel = img.ravel()
    if drop != None:
        img_ravel = img_ravel[img_ravel!=drop]
    plt.hist(img_ravel, bins=100)
    plt.show()
    
def enhance(img, brightness, contrast):
    """Help the user to increase an image brightness and contrast without overflowing the memory for display purposes.

    Arguments:
    -----------
        img: np.ndarray
            Image that will be enhanced
        brightness: float
            Range from 0 to 1, 0 means the image is kept as is, 1 means the entire image pixels are at maximal value.
        brightness: float
            Range from 0 to 1, 0 means the image is kept as is, 1 means the entire image pixels are either at maximal or minimal value.
    Returns:
    -----------
        img: np.ndarray
            Initial image with increased brightness and contrast
    """
    IINFO = np.iinfo(img.dtype)

    if brightness != 0:
        brightness = np.floor((IINFO.max-IINFO.min)*brightness)

        alpha = (IINFO.max - np.abs(brightness)) / IINFO.max
        gamma = np.max(brightness, 0)

        img = img.copy()*alpha + gamma
        img[img>IINFO.max] = IINFO.max
        img[img<IINFO.min] = IINFO.min
        img = img.astype(IINFO.dtype)

    if contrast != 0:
        contrast = np.floor((IINFO.max-IINFO.min)*contrast)

        HALF = np.floor((IINFO.max+IINFO.min)/2)
        HALF_1 = HALF + 1

        f = HALF_1*(contrast+HALF)/(HALF_1-contrast)
        alpha = f/HALF
        gamma = HALF - f

        img = img.copy()*alpha + gamma
        img[img>IINFO.max] = IINFO.max
        img[img<IINFO.min] = IINFO.min
        img = img.astype(IINFO.dtype)
    
    return img
    
def plot(imgs, subplots_shape, figsize1=30, figsize2=30, brightness=0, contrast=0, colorbar=False, titles=None):
    """Multipurpose image plotting framework. Expects an image of a list of images as input (following the cv2 style: BGR, [y coordinates, x coordinates, bands]).
    In case a list of images is provided, all images will be displayed on subplots.


    Arguments:
    -----------
        img: list
            Image being displayed, should follow cv2 imae format style: BGR, [y coordinates, x coordinates, bands]
        subplots_shape: list
            list of size 2. First value of the list will define the number of rows in of the subplot while second value defines the number of columns.
        figsize: int
            Defined img sizes, usual values I use are ranging from 10 to 100
        brightness: float
            Increases brigthness of BGR images only using the 'enhance' function.
        contrast: float
            Increases brigthness of BGR images only using the 'enhance' function.
        colorbar: boolean
            If True a colorbar will be displayed   
        titles: list
            list of a size equal to the size of 'img'. Respective 'titles' will be displayed on top of the plots
    """
    
    # In case single image, convert to list
    if type(imgs) != list:
        imgs = [imgs]

    # Making sure the caneva make sense
    if subplots_shape == None:
        subplots_shape = [1, len(imgs)]

    # Initiating subplots
    fig,axes  = plt.subplots(nrows = subplots_shape[0], 
                             ncols = subplots_shape[1], 
                             figsize=(figsize1,figsize2), 
                             squeeze=True)

    # Filling subplots
    i = 0
    for x in range(subplots_shape[0]):
        for y in range(subplots_shape[1]):

            img = imgs[i].copy()

            # In case 3 band img, bgr to rgb
            if (len(imgs[i].shape) > 2) & (imgs[i].shape[-1] == 3):
                img[:, :, 0] = imgs[0][:, :, 2].copy()
                img[:, :, 2] = imgs[0][:, :, 0].copy()

                # If Enhance
                img = enhance(img, brightness=brightness, contrast=contrast)

            # Plot
            a = axes[x, y].imshow(img, cmap=plt.cm.gray)
            axes[x, y].axis('off')
            axes[x, y].set_title(titles[i] if titles is not None else str(i))
            if colorbar:
                plt.colorbar(a,ax=axes[x,y])

            i+=1

    plt.show()   
    
    
def plus(a, b):
    """Add an np.ndarray 'b' to an np.ndarray 'a'. 'plus' does it in an efficient way and also avoids memory overflow by clipping to 'a' dtype. 

    Arguments:
    -----------
        a: np.ndarray
            Reference array.
        b: np.ndarray
            b is added to a, its dtype can be different but his shape should be identical.
        
    Returns:
    -----------
        a: np.ndarray
            a + b
    """
    
    assert a.dtype == b.dtype, "a & b have different types"
    
    MAX = np.iinfo(a.dtype).max
    
    b = MAX - b  # old b is gone shortly after new array is created
    np.putmask(a, b < a, b)  # a temp bool array here, then it's gone
    a = a + MAX - b  # a temp array here, then it's gone
    
    return a


def minus(a, b):
    """Substracts an np.ndarray 'b' to an np.ndarray 'a'. 'minus' does it in an efficient way and also avoids memory overflow by clipping to 'a' dtype. 

    Arguments:
    -----------
        a: np.ndarray
            Reference array.
        b: np.ndarray
            b is substracted to a, its dtype can be different but his shape should be identical.
        
    Returns:
    -----------
        a: np.ndarray
            a - b
    """
    
    assert a.dtype == b.dtype, "a & b have different types"
    
    MIN = np.iinfo(a.dtype).min
    
    b = b + MIN  # old b is gone shortly after new array is created
    np.putmask(a, a < b, b)  # a temp bool array here, then it's gone
    a = a - b - MIN  # a temp array here, then it's gone
    
    return a
    
    
def hist_match(source, template, dtype=None):
    """Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts)
    s_quantiles = s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts)
    t_quantiles = t_quantiles / t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values_reshaped =  interp_t_values[bin_idx].reshape(oldshape)
    
    # Type conversion
    if dtype:
        interp_t_values_reshaped = interp_t_values_reshaped.astype(dtype)

    return interp_t_values_reshaped