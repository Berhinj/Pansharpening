#! /usr/bin/env python
"""Quality Index Calculation Functions.

This scripts group several common spectral distortion index that will be usefull when assessing a pansharpening algorithm.
Note that these functions expects stacks of 1D bands so images need to be ravel/reshaped.
"""

import numpy as np


def RMSEs(A, B):
    """RMSE stands for Root Mean Square Error. In this case, the function computes the RMSEs band per band between two band stacks.
    RMSE ranges from 0 to 1, the lower the RMSE, the lower the difference between the two stack of array. It will be used for futher computations.
    
    Arguments:
    -----------
        A: np.ndarray
            Stack of 1D bands
        B: np.ndarray
            Stack of 1D bands
           
    Returns:
    -----------
        RMSEs: np.array
            1D np array with one RMSE value per bands
    """
    A = A.astype("float64")
    B = B.astype("float64")
    
    # Squared Error
    SE = (B - A)**2
    
    # Root Mean Square Error
    RMSEs = np.average(SE, 0) ** 0.5
    
    return RMSEs


def ERGAS(A, B, RMSEs, f_ratio):
    """ERGAS stands for Relative Dimensionless Global Error in Synthesis. Common index employed to measure the spectral distortion between the fused and original MS bands.
    Improved version of the index named Relative Average Spectral Error (RASE), which is defined based on Root Mean Square Error (RMSE).
    ERGAS ranges from 0 to 100, the lower the ERGAS, the lower the difference between the two stack of array.
    
    Li, H., Jing, L., & Tang, Y. (2017). Assessment of Pansharpening Methods Applied to WorldView-2 Imagery Fusion. Sensors (Basel, Switzerland), 17(1), 89. 
    https://doi.org/10.3390/s17010089
    
    Arguments:
    -----------
        A: np.ndarray
            Stack of 1D bands
        B: np.ndarray
            Stack of 1D bands
        RMSE: float
            Root Mean Square Error
        f_ratio: float
            sqrt(Pansharen Band Resolution/Initial Band Resolution). 
            Example: PAN 6x6 pixel & Other bands 3x3 bands => sqrt('6x6'/'3x3') = sqrt(36/9) = sqrt(4) => f_ratio = 2
           
    Returns:
    -----------
        ERGAS: float
            ERGAS index
    """
    A = A.astype("float64")
    B = B.astype("float64")
    
    # Mean
    Ms = np.average(B, 0)
    
    # Relative Dimensionless Global Error in Synthesis
    ERGAS = 100 * f_ratio**2 * (np.sum((RMSEs/Ms)**2)/A.shape[1])**0.5
    
    return ERGAS


def RASE(A, B, RMSEs):
    """RASE stands for Relative Average Spectral Error, it is a modified on Root Mean Square Error (RMSE) to make it suited for multiple bands.
    Common index employed to measure the spectral distortion between the fused and original MS bands.
    RASE ranges from 0 to 100, the lower the ERGAS, the lower the difference between the two stack of array.
    
    Wald L. Quality of high resolution synthesised images: Is there a simple criterion?; Proceedings of the International Conference on Fusion Earth Data; 
    Sophia Antipolis, France. 26–28 January 2000 
    https://hal.archives-ouvertes.fr/hal-00395027/document
    
    Arguments:
    -----------
        A: np.ndarray
            Stack of 1D bands
        B: np.ndarray
            Stack of 1D bands
        RMSE: float
            Root Mean Square Error
           
    Returns:
    -----------
        RASE: float
            RASE index
    """
    A = A.astype("float64")
    B = B.astype("float64")
    
    # Mean
    Ms = np.average(A, axis=0)
    
    # Relative Average Spectral Error
    RASE = np.average(100*np.sqrt(RMSEs**2/A.shape[1])/Ms)
    
    return RASE


def SAM(A, B):
    """SAM stands for Spectral Angle Mapper. Common index employed to measure the spectral distortion between the fused and original MS bands with the average spectral angle of all pixels involved.
    The lower the SAM, the lower the difference between the two stack of array.
    
    Wald L. Quality of high resolution synthesised images: Is there a simple criterion?; Proceedings of the International Conference on Fusion Earth Data; Sophia Antipolis, 
    France. 26–28 January 2000
    https://hal.archives-ouvertes.fr/hal-00395027/document
    
    Arguments:
    -----------
        A: np.ndarray
            Stack of 1D bands
        B: np.ndarray
            Stack of 1D bands
           
    Returns:
    -----------
        SAM: float
            SAM index
    """
    A = A.astype("float64")
    B = B.astype("float64")
    
    # Spectral Angle Mapper   
    sum_AB = np.sum(A*B, 1)
    sum_A = np.sum(A, 1)
    sum_B = np.sum(B, 1)
    SAMs = np.arccos(np.clip(sum_AB/(sum_A * sum_B),0,1))
    SAM = np.average(SAMs)*180/np.pi
    
    return SAM


def Q2N(A, B):
    """Q2N stands for Universal Image Quality Index (Generalization of Q index for monoband images). Common index employed to measure the spectral distortion between the fused and original MS bands.
    The lower the Q2N, the lower the difference between the two stack of array.
    
    Li, H., Jing, L., & Tang, Y. (2017). Assessment of Pansharpening Methods Applied to WorldView-2 Imagery Fusion. Sensors (Basel, Switzerland), 17(1), 89. 
    https://doi.org/10.3390/s17010089
    
    Arguments:
    -----------
        A: np.ndarray
            Stack of 1D bands
        B: np.ndarray
            Stack of 1D bands
           
    Returns:
    -----------
        Q2N: float
            Q2N index
    """
    A = A.astype("float64")
    B = B.astype("float64")
    
    # Universal Image Quality Index (Q-average)
    mean_A = np.average(A, axis=0)
    mean_B = np.average(B, axis=0)
    covar_AB = np.average((A-mean_A)*(B-mean_B), axis=0)
    var_A = np.var(A, axis=0)
    var_B = np.var(B, axis=0)
    Q2N = np.average((4*covar_AB*mean_A*mean_B)/((var_A+var_B)*(mean_A**2 + mean_B**2)))
    
    return Q2N


