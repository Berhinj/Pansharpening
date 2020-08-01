#! /usr/bin/env python
""""""

import quality

import cv2
import numpy as np


def evaluate(BS, PAN, f_ratio_evaluate, pansharpening_function, **kwargs):
    """Wald Second Evaluation Framework.

    These functions offers some accelerators to set up Wald Second Evalution Method (Pushparaj et al. 2017). 
    This method suggest a solution to create quantitative index to assess a pansharpening method without groundtruth. 
    The logic is the following one: Use raw multispectral bands as groundtruth and compare then to a panned downscaled version of itself. 

    The quality indexes are then being computed based on: On one side the initial BS bands and on the otherside the downscaled panned BS bands.
    For quality indexes, the lower the beter.
    
    Pushparaj, J., Hegde, A.V. Evaluation of pan-sharpening methods for spatial and spectral quality. Appl Geomat 9, 1–12 (2017). https://doi.org/10.1007/s12518-016-0179-2

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        f_ratio_evaluate: float
            sqrt(Pansharen Band Resolution/Initial Band Resolution). Equal to f_ratio. Naming is f_ratio_evaluate instead of f_ratio because f_ratio can be part of **kwargs
            Example: PAN 6x6 pixel & Other bands 3x3 bands => sqrt('6x6'/'3x3') = sqrt(36/9) = sqrt(4) => f_ratio = 2
        pansharpening_function: function
            pansharpening function applied to the downscaled BS bands
        **kwargs:
            parameters that need to be passed to the pansharpening function
        
    Returns:
    -----------
        results: dictionnary
            Dictionnary grouping all quality indexes
    """

    # Downscaling BS with linear interpolation twice
    BS_2xdownscaled = cv2.resize(BS, None, fx=1/(f_ratio_evaluate**2), fy=1/(f_ratio_evaluate**2), interpolation=cv2.INTER_LINEAR)
    
    # Upscaling BS with nearest interpolation
    BS_2xdownscaled_1xupscaled = cv2.resize(BS_2xdownscaled, None, fx=f_ratio_evaluate, fy=f_ratio_evaluate, interpolation=cv2.INTER_NEAREST)
    PAN_1xdownscaled = cv2.resize(PAN, None, fx=1/f_ratio_evaluate, fy=1/f_ratio_evaluate, interpolation=cv2.INTER_LINEAR)
    
    # Pansharpening downscaled bands
    BS_downscaled_panned = pansharpening_function(
        BS = BS_2xdownscaled_1xupscaled, 
        PAN = PAN_1xdownscaled, 
        **{key:kwargs[key] for key in kwargs if key not in ["BS", "PAN"]})
    
    BS = cv2.resize(BS, None, fx=1/f_ratio_evaluate, fy=1/f_ratio_evaluate, interpolation=cv2.INTER_NEAREST)
    
    # Quality Indexes calculation (need reshaping)
    reshaping = [BS.shape[0]*BS.shape[1], BS.shape[2]]
    BS = np.reshape(BS, reshaping).astype(BS.dtype)
    BS_downscaled_panned = np.reshape(BS_downscaled_panned, reshaping).astype(BS.dtype)
    
    results = {}
    results["RMSEs"] = quality.RMSEs(BS, BS_downscaled_panned)
    results["ERGAS"] = quality.ERGAS(BS, BS_downscaled_panned, RMSEs=results["RMSEs"], f_ratio=f_ratio_evaluate)
    results["RASE"] = quality.RASE(BS, BS_downscaled_panned, RMSEs=results["RMSEs"])
    results["SAM"] = quality.SAM(BS, BS_downscaled_panned)
    results["Q2N"] = quality.Q2N(BS, BS_downscaled_panned)
    
    return results
 