import numpy as np
from scipy import signal, ndimage
from scipy.interpolate import interp2d
from sklearn.preprocessing import StandardScaler

def load_2d_nmr(data_path):
    """
    Load 2D NMR data from file
    Assumes data is in numpy format (.npy) or similar
    """
    return np.load(data_path)

def remove_noise(spectrum, sigma=1.0):
    """
    Reduce noise using Gaussian filtering
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    sigma: Standard deviation for Gaussian kernel
    
    Returns:
    Denoised spectrum
    """
    return ndimage.gaussian_filter(spectrum, sigma=sigma)

def correct_baseline(spectrum, polynomial_order=3):
    """
    Perform baseline correction using polynomial fitting
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    polynomial_order: Order of polynomial for fitting
    
    Returns:
    Baseline corrected spectrum
    """
    rows, cols = spectrum.shape
    corrected = np.zeros_like(spectrum)
    
    # Correct baseline for each row and column
    for i in range(rows):
        x = np.arange(cols)
        coeffs = np.polyfit(x, spectrum[i,:], polynomial_order)
        baseline = np.polyval(coeffs, x)
        corrected[i,:] = spectrum[i,:] - baseline
    
    for j in range(cols):
        x = np.arange(rows)
        coeffs = np.polyfit(x, corrected[:,j], polynomial_order)
        baseline = np.polyval(coeffs, x)
        corrected[:,j] = corrected[:,j] - baseline
    
    return corrected

def align_spectra(reference, target, max_shift=20):
    """
    Align target spectrum to reference spectrum using cross-correlation
    
    Parameters:
    reference: 2D numpy array of reference spectrum
    target: 2D numpy array of spectrum to be aligned
    max_shift: Maximum allowed shift in pixels
    
    Returns:
    Aligned spectrum
    """
    # Compute 2D cross-correlation
    correlation = signal.correlate2d(reference, target, mode='same')
    
    # Find shift that gives maximum correlation
    y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    y_shift -= reference.shape[0] // 2
    x_shift -= reference.shape[1] // 2
    
    # Limit shifts to max_shift
    y_shift = np.clip(y_shift, -max_shift, max_shift)
    x_shift = np.clip(x_shift, -max_shift, max_shift)
    
    # Apply shift using interpolation
    y = np.arange(target.shape[0])
    x = np.arange(target.shape[1])
    f = interp2d(x, y, target)
    
    y_new = y - y_shift
    x_new = x - x_shift
    aligned = f(x_new, y_new)
    
    return aligned

def normalize_spectrum(spectrum, method='standard'):
    """
    Normalize spectrum using various methods
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    method: Normalization method ('standard', 'minmax', or 'total')
    
    Returns:
    Normalized spectrum
    """
    if method == 'standard':
        # Standardize to zero mean and unit variance
        scaler = StandardScaler()
        flat_spectrum = spectrum.reshape(-1, 1)
        normalized_flat = scaler.fit_transform(flat_spectrum)
        return normalized_flat.reshape(spectrum.shape)
    
    elif method == 'minmax':
        # Scale to range [0,1]
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        return (spectrum - min_val) / (max_val - min_val)
    
    elif method == 'total':
        # Normalize by total intensity
        return spectrum / np.sum(spectrum)
    
    else:
        raise ValueError("Unknown normalization method")

def preprocess_2d_nmr(spectrum, noise_sigma=1.0, baseline_order=3, 
                     reference=None, max_shift=20, norm_method='standard'):
    """
    Complete preprocessing pipeline for 2D NMR spectrum
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    noise_sigma: Sigma for noise reduction
    baseline_order: Order of polynomial for baseline correction
    reference: Reference spectrum for alignment (optional)
    max_shift: Maximum allowed shift for alignment
    norm_method: Normalization method
    
    Returns:
    Preprocessed spectrum
    """
    # Noise reduction
    spectrum_denoised = remove_noise(spectrum, sigma=noise_sigma)
    
    # Baseline correction
    spectrum_baselined = correct_baseline(spectrum_denoised, 
                                        polynomial_order=baseline_order)
    
    # Alignment (if reference provided)
    if reference is not None:
        spectrum_aligned = align_spectra(reference, spectrum_baselined, 
                                       max_shift=max_shift)
    else:
        spectrum_aligned = spectrum_baselined
    
    # Normalization
    spectrum_normalized = normalize_spectrum(spectrum_aligned, 
                                          method=norm_method)
    
    return spectrum_normalized

# Example usage
if __name__ == "__main__":
    # Load data
    spectrum = load_2d_nmr("path_to_spectrum.npy")
    reference = load_2d_nmr("path_to_reference.npy")
    
    # Preprocess spectrum
    processed_spectrum = preprocess_2d_nmr(
        spectrum,
        noise_sigma=1.0,
        baseline_order=3,
        reference=reference,
        max_shift=20,
        norm_method='standard'
    )