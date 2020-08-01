#! /usr/bin/env python
"""Pansharpening Functions.

This script groups all implemented pansharpening technics. 
WARNING: All pansharpening functions expects:
PAN: Panchromatic Band. 
    It can be the actual panchromatic band or a simulated one whatever, as long as its shape is [y dimension, x dimension].
    This shape is the default opencv/matplotlib single band image format. It's resolution is expected to be the original one.
BS: Spectral Band(s) to be pansharpen. 
    Its expected shape is [y dimension, x dimension, band dimension]. This shape is the default opencv multiband band format.
    It's resolution is EXPECTED TO BE UPSCALED TO PAN RESOLUTION (I would advise aligning BS with PAN first then using cv2.resize function with interpolation=cv2.INTER_NEAREST).S
"""

from utils import plus, minus, hist_match
import cv2
import numpy as np
import pywt
from sklearn import decomposition
import wald



class PansharpeningStructure():   
    """Parent class for pansharpening transformations"""        
    
    def transform(self):
        """Runs the pansharpening method"""
        
        self.BS_panned = self.pansharpening(**self.pansharpening_param)
        return self.BS_panned
    
            
    def evaluate(self):
        """Runs wald evaluation"""
        
        self.results = wald.evaluate(
            f_ratio_evaluate = self.f_ratio,
            pansharpening_function=self.pansharpening,
            **self.pansharpening_param)
    
        return self.results
 

class Ihs(PansharpeningStructure):
    """The IHS pansharpening method is the most basic pansharpening approach. This is a space conversion method, which means the bands are converted to
    a new spectral space where bands are (supposedly) decorrelated. In this case, the spectral space is IHS (Intensity, Hue & Saturation).
    In the IHS space, Intensity is expected to be the 'spatial' dimension of the bands, while Hue & Saturation would be representing the 'colored' dimensions.
    Converting the bands to IHS is called the forward transformation. Once the forward tranformation is done, the 'spatial' band (I, Intensity) is manually replaced
    by the PAN band also representing the spatial 'factor' of the bands. Then a backward transformation is performed to revert back to the usual color space. 
    Per design IHS method is limited to 3 bands.

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 3x2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        hist_matching: bool, default=True
            If True, the histogram of the PAN band will be matched to the intensity of IHS before proceeding to the backward transformation.
        f_ratio: float
            sqrt(Pansharen Band Resolution/Initial Band Resolution). 
            Example: PAN 6x6 pixel & Other bands 3x3 bands => sqrt('6x6'/'3x3') = sqrt(36/9) = sqrt(4) => f_ratio = 2

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
    
    def __init__(self, BS, PAN, f_ratio, hist_matching=True):
        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
        
    def pansharpening(self, BS, PAN, hist_matching):
        """Pansharpening algorithm itself"""

        # Convert to IHS/HSV color space
        HSV = cv2.cvtColor(BS, cv2.COLOR_RGB2HSV)

        # Replacing intensity band (Histogram matching before replacing intensity band)
        HSV[:, :, 2] = hist_match(PAN, HSV[:, :, 2], dtype = PAN.dtype) if hist_matching else PAN 

        # Revert to original color space
        BS_panned = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)

        return BS_panned
    
    
class Gihs(PansharpeningStructure):
    """GIHS is a Generalized IHS pansharpening method so it can work with a number of bands superior to 3.
    It basically works in the same than IHS does but used some shortcuts in the calculation to make it more efficient and functionnal on more than 3 bands.
    Theory being used for this implementation comes from De Luciano Alparone et al. 2015. For efficiency purposes, the algorithm uses custom substraction 
    function to make sure the memory isn't overflown or convert to a different type. 

    De Luciano Alparone et al. 2015, Remote Sensing Image Fusion, CRC Press - PAGE 132-133: 6.2.3.1 Linear IHS
    https://books.google.be/books?id=sXgZBwAAQBAJ&pg=PA134&lpg=PA134&dq=rgb+to+ihs+matrix+multiplication&source=bl&ots=yXcYKuR8te&sig=ACfU3U3w3SysngFj0AwVYyyZ_WRc7KsbTw&hl=fr&sa=X&ved=2ahUKEwj86pWlq-LoAhUPMewKHVZVCxcQ6AEwAXoECAoQAQ#v=onepage&q=rgb%20to%20ihs%20matrix%20multiplication&f=false

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        hist_matching: bool, default=True
            If True, the histogram of the PAN band will be matched to the intensity of IHS before proceeding to the backward transformation.

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
        
    def __init__(self, BS, PAN, f_ratio, hist_matching=True):
        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
       
        
    def pansharpening(self, BS, PAN, hist_matching=True):
        """Pansharpening algorithm itself"""

        # No need to convert IHS/HSV color space - Direct Intensity Band swap
        intensity = np.mean(BS, axis=-1, dtype=BS.dtype)

        # Histogram matching before replacing intensity band
        if hist_matching:
            PAN = hist_match(PAN, intensity, dtype = PAN.dtype)

        # Replacing intensity band - Using custom function to avoid overflow and keep performance high
        BS_panned = minus(plus(BS, np.expand_dims(PAN, axis=-1)), np.expand_dims(intensity, axis=-1))

        return BS_panned
    
    
class Pca(PansharpeningStructure):
    """The PCA pansharpening method, is also space conversion method. The stack of images bands are converted to components, the first component, 
    assumed to represent the spatial dimension of the component bands (per construction) is then replaced by the PAN band and a backward transformation
    is finally performed. The bands resulting of the PCA backward transformation are the BS bands panned.

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        hist_matching: bool, default=True
            If True, the histogram of the PAN band will be matched to the first component of the PCA before proceeding to the backward transformation.

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
        
    def __init__(self, BS, PAN, f_ratio, hist_matching=True):
        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
       
        
    def pansharpening(self, BS, PAN, hist_matching=True):
        """Pansharpening algorithm itself"""
        # Reshaping into a 1D arrays
        BS_reshape = np.reshape(BS, (BS.shape[0] * BS.shape[1], BS.shape[2]))
        PAN = np.reshape(PAN, BS.shape[0] * BS.shape[1])

        # PCA Decomposition
        model = decomposition.PCA(BS.shape[2])
        BS_components = model.fit_transform(BS_reshape)

        # Histogram matching before replacing first component
        if hist_matching:
            PAN = hist_match(PAN, 
                             BS_components[:, 0], 
                             dtype = BS_components.dtype)

        # Replacing the PCA first component
        BS_components[:, 0] = PAN

        # Invecting PCA
        BS_panned = model.inverse_transform(BS_components)

        # Reshaping into 2D arrays 
        BS_panned = np.reshape(BS_panned, BS.shape).astype(BS.dtype)
        
        return BS_panned
    
    
class Brovey(PansharpeningStructure):
    """'Brovey uses a method that multiplies each resampled, multispectral pixel by the ratio of the corresponding panchromatic pixel intensity to the sum of all the multispectral intensities. 
    It assumes that the spectral range spanned by the panchromatic image is the same as that covered by the multispectral channels.'
    https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        weights: list
            list of weights applied for the linear combinations of the multispectral images

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
        
    def __init__(self, BS, PAN, weights, f_ratio):
        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
       
        
    def pansharpening(self, BS, PAN, weights):
        """Pansharpening algorithm itself"""
        # Assessing & cleaning weigths
        assert len(weights) == BS.shape[-1], "'weights' length doesn't align with the number of band in 'BS'"
        weights = weights/np.sum(weights)

        # Brovey Transformation
        multiplicator = PAN / np.sum(BS * weights, 2)

        BS_panned = np.zeros(BS.shape, dtype = "uint8")
        for i in range(BS.shape[-1]):
            BS_panned[:, :, i] = (BS[:, :, i] * multiplicator)
            
        return BS_panned
    
    
class Gs(PansharpeningStructure):
    """As explained by Laben et al. (2000), 'The Spatial resolution of a multispectral digital image is enhanced in a process of the type
    wherein a higher spatial resolution panchromatic image is merged with a plurality of lower Spatial resolution Spectral band images.
    A lower Spatial resolution panchromatic image is simulated and a Gram-Schmidt (GS) transformation is performed on the simulated lower 
    spatial resolution panchromatic image and the plural ity of lower spatial resolution spectral band images. The Simulated lower 
    Spatial resolution panchromatic image is employed as the first band in the Gram-Schmidt transformation.' 

    This script is following the methodology described in the patent (now expired since 2019), with additionnal tweaks coming Maurer (2000), 
    regarding the simulated PAN band & the orthogonalization process by removing the average term.

    Original Patent Method: Laben et al. 2000, United States Patent, PROCESS FOR ENHANCING THE SPATIAL
    RESOLUTION OF MULTISPECTRAL IMAGERY USING PAN-SHARPENING, 
    https://patentimages.storage.googleapis.com/f9/72/45/c9f1fffe687d30/US6011875.pdf

    Additionnal tweaks from: Maurer 2000, International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences,
    Volume XL-1/W1, ISPRS Hannover Workshop 2013, 21 – 24 May 2013, Hannover, Germany, 
    https://www.researchgate.net/publication/274676820_How_to_pan-sharpen_images_using_the_Gram-Schmidt_pan-sharpen_method_-_A_recipe

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp band; has to have same dimension as BS bands
        f_ratio: float
            sqrt(Pansharen Band Resolution/Initial Band Resolution). 
            Example: PAN 6x6 pixel & Other bands 3x3 bands => sqrt('6x6'/'3x3') = sqrt(36/9) = sqrt(4) => f_ratio = 2
        PANsim_method: string {'linear', 'combination'}, default='linear'
            Method used to simulate the low resolution PAN band required for Gram-Schmidt forward transform. If 'linear', 
            then the simulated PAN band will be based on a resampling using a linear interpolation of the initial PAN band,
            as suggested in the Original Patent Method. If 'combination', then the simulated PAN band will be based on a linear combination
            of BS bands as explained in Maurer 2000.
        hist_matching: bool, default=False
            If True, the histogram of the PAN band (used in the backward GS Transformation) will be matched to the simulated PAN band (used
            in the forward GS Transformation) 

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
        
    def __init__(self, BS, PAN, f_ratio, PANsim_method="linear", hist_matching=False):        

        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["self"]}
        self.f_ratio = f_ratio 
        
        
    def pansharpening(self, BS, PAN, f_ratio, PANsim_method="linear", hist_matching=False):
        """Pansharpening algorithm itself"""
    
        def cov2(A, B):
            """Covariance calculation between np.ndarray A and B
            """

            return np.average((A-np.average(A))*(B-np.average(B)))

        def ravels(img):
            """'Ravels' every 2D imgage of an image stack seperatly as would np.ravel
            """
            return np.reshape(img, [img.shape[0]*img.shape[1], img.shape[2]]).T

        # Downscaling MS => ms
        bs = BS.copy()

        # Downscaling PAN => pan
        if PANsim_method == 'combination':
            PANblured = cv2.filter2D(PAN,-1, np.ones((3,3))/9)
            BS_PANblured = np.concatenate([np.expand_dims(PANblured, -1), BS], -1)
            C_PANblured = np.cov(ravels(BS_PANblured))

            wk = np.sum(np.linalg.inv(C_PANblured[1:, 1:]), 1)*C_PANblured[1:, 0]
            pan = np.sum(wk * bs, -1)

        else:
            pan = cv2.resize(PAN, None, fx=1/f_ratio, fy=1/f_ratio, interpolation=cv2.INTER_LINEAR)
            pan = cv2.resize(pan, None, fx=f_ratio, fy=f_ratio, interpolation=cv2.INTER_NEAREST)

            if PANsim_method != 'linear':
                print("PANsim_method {s} is not valid, Default PAN Simulated Method used: 'linear' down sampling".format(PANsim_method))

        bs_pan = np.concatenate([np.expand_dims(pan, -1), bs], -1)

        # Histogram Matching
        if hist_matching:
            PAN = hist_match(PAN, pan, dtype = PAN.dtype)

        # Gram-Schmidt Orthogonalization : initialization
        GS = np.zeros(bs_pan.shape[-1])
        GS_BS = np.zeros(bs_pan.shape)
        GS_bs = np.zeros(bs_pan.shape)
        BS_panned = np.zeros(bs_pan.shape)
        bs_rot = np.zeros(bs_pan.shape)

        # Gram-Schmidt Orthogonalization : Forward Transformation
        for i in range(bs_pan.shape[-1]):
            GS[i] = cov2(bs_pan[:, :, i], bs_rot[:, :, i-1])/np.var(bs_rot[:, :, i-1]) if i>0 else 0

            GS_bs[:, :, i] = GS[i]*bs_rot[:, :, i-1]

            bs_rot[:, :, i] = bs_pan[:, :, i] - np.sum(GS_bs, -1) # Swapping PAN

        # Gram-Schmidt Orthogonalization: Backward Transformation
        bs_rot[:, :, 0] = PAN
        for i in range(bs_pan.shape[-1]):

            GS_BS[:, :, i] = GS[i]*bs_rot[:, :, i-1]
            BS_panned[:, :, i] = bs_rot[:, :, i] + np.sum(GS_BS, -1)

        return BS_panned[:, :, 1:].astype(BS.dtype)
    
    
class Wavelet(PansharpeningStructure):
    """The Wavelet pansharpening method, is a space conversion method based on IHS. The stack of images bands are converted to IHS, 
    the intensity band is fusionned with the PAN band using a wavelet image fusion method. The IHS image is then tranformed back to regular color bands.

    The wavelet image fusion method consists in a wavelet decomposition of both I and PAN. Then the lower resolution images resulting
    from the decomposition of both PAN and I are merged following a specific strategy called here 'Fusion Method'. Penty of 'Fusion Method' exists,
    however not all of them have been implemented here already. Only the basics are available a the moment. 

    To get which 'wavelet' type, decomposition 'mode' and 'fusion' method are available, use 'wavelet_help()'.

    Arguments:
    -----------
        BS: np.ndarray
            Stack of 2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands
        hist_matching: bool, default=True
            If True, the histogram of the PAN band will be matched to the first component of the PCA before proceeding to the backward transformation.
        mode: string
            wavelet decomposition mode. More info in pywt.wavedec2 documentation.
        wavelet: string
            wavelet shape. More info in pywt.wavedec2 documentation.
            level: integer
            decomposition iteration count. If None, the image will be fully decomposed (heavy computation)
        fusion: string
            Fusion calculation method being used between B_coef and PAN_coef

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
    
    def __init__(self, BS, PAN, f_ratio, hist_matching=True, mode='symmetric', wavelet="haar", level=None, fusion="max"):        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
       
        
    def pansharpening(self, BS, PAN, hist_matching=True, mode='symmetric', wavelet="haar", level=None, fusion="max"):
        """Pansharpening algorithm itself"""
        
        def coef_fuse(B_coef, PAN_coef, fusion):
            """Coeficient 'fusion' calculation. The calculation depends mainly on the 'fusion' method selected.
            To get which 'fusion' method are available, use 'wavelet_help()'

            Arguments:
            -----------
                B_coef: np.ndarray
                    Band decomposed coefficient
                PAN_coef: np.ndarray
                    Pansharpening Band decomposed coefficient
                mode: string
                    wavelet decomposition mode. More info in pywt.wavedec2 documentation.
                wavelet: string
                    wavelet shape. More info in pywt.wavedec2 documentation.
                level: integer
                    decomposition iteration count. If None, the image will be fully decomposed (heavy computation)
                fusion: string
                    Fusion calculation method being used between B_coef and PAN_coef

            Returns:
            -----------
                coef: np.ndarray
                    Fusionned coefficient
            """

            if (fusion == 'mean'):
                coef = np.average((B_coef, PAN_coef), axis=0)
            elif (fusion == 'min'):
                coef = np.minimum(B_coef, PAN_coef)
            elif (fusion == 'max'):
                coef = np.maximum(B_coef, PAN_coef)
            elif (fusion == 'band'):
                coef = B_coef,
            elif (fusion == 'pan'):
                coef = PAN_coef
            else:
                raise ValueError("Fusion method doesn't exists.")

            return tuple(coef)


        def wavelet_fusion(A, B, mode='symmetric', wavelet="haar", level=None, fusion = "max"):
            """Wavelet decomposition creates several lower resolution image based on the input image.  fusionning 
            Wavelet Image Fusion consist in decomposing two image according to wavelet decomposion algorithm and sufionning respective lower resolution images,
            before reverting the decomposition.

            Arguments:
            -----------
                A: np.ndarray
                    2D Image
                B: np.ndarray
                    2D Image
                mode: string
                    wavelet decomposition mode. More info in pywt.wavedec2 documentation.
                wavelet: string
                    wavelet shape. More info in pywt.wavedec2 documentation.
                level: integer
                    decomposition iteration count. If None, the image will be fully decomposed (heavy computation)
                fusion: string
                    Fusion calculation method being used between B_coef and PAN_coef

            Returns:
            -----------
                AB_fused: np.ndarray
                    A & B image fusionned
            """
            # Wavelet Decomposition
            A_dec = pywt.wavedec2(data=A, mode=mode, wavelet=wavelet, level=level)
            B_dec = pywt.wavedec2(data=B, mode=mode, wavelet=wavelet, level=level)

            # Fusing coefficient
            coef_fused = [coef_fuse(A_coef, B_coef, fusion) for A_coef, B_coef in zip(A_dec, B_dec)]

            # Inverse Wavelet Decomposition
            AB_fused = pywt.waverec2(coef_fused, wavelet)

            return AB_fused


        # Limited to 3 bands
        assert BS.shape[-1] == 3, "BS should contain exactly 3 bands"

        # Convert to IHS/HSV color space
        HSV = cv2.cvtColor(BS, cv2.COLOR_RGB2HSV)

        # Histogram matching before replacing intensity band
        if hist_matching:
            PAN = hist_match(PAN, HSV[:, :, 2], dtype = PAN.dtype)

        # Replacing intensity band with Wavelet Fusion
        HSV[:, :, 2] = wavelet_fusion(HSV[:, :, 2], PAN).astype(BS.dtype)

        # Revert to original color space
        BS_panned = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)

        return BS_panned


class DummyPansharpening(PansharpeningStructure):
    """The Dummy pansharpening method exists for the same reason than DummyClassifier/DummyRegressor exist in Scikit Learn package.
    It pansharpen images using simplitic rules. It is useful in the sense that it creates a baseline to compare models performances.
    Arguments:
    -----------
        BS: np.ndarray
            Stack of 3x2D bands to pansharpen (already resampled to PAN band resolution)
        PAN: np.ndarray
            2D Pansharp image; has to have same dimension as BS bands

    Returns:
    -----------
        BS_panned: np.ndarray
            BS bands but panned
    """
    
    def __init__(self, BS, PAN, f_ratio):
        
        self.__dict__.update(locals())
        self.pansharpening_param = {key:self.__dict__[key] for key in self.__dict__ if key not in ["f_ratio", "self"]}
        self.f_ratio = f_ratio
        
    def pansharpening(self, BS, PAN):
        """Pansharpening algorithm itself"""

        BS_panned = BS.copy()

        return BS_panned