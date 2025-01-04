import numpy as np
from scipy import signal, ndimage
from sklearn.feature_extraction import image
from scipy.stats import skew, kurtosis
import pandas as pd

def find_peaks_2d(spectrum, height_threshold=0.1, min_distance=5, 
                  prominence=0.05, width=None):
    """
    Detect peaks in 2D NMR spectrum using local maxima detection
    
    Parameters:
    spectrum: 2D numpy array of processed spectral data
    height_threshold: Minimum peak height relative to maximum intensity
    min_distance: Minimum distance between peaks
    prominence: Required peak prominence
    width: Optional peak width constraints
    
    Returns:
    peaks_df: DataFrame with peak properties
    """
    # Find local maxima
    peaks = signal.peak_local_max(
        spectrum,
        min_distance=min_distance,
        threshold_rel=height_threshold,
        prominence=prominence,
        width=width
    )
    
    # Extract peak properties
    peak_properties = []
    for y, x in peaks:
        # Get peak intensity
        intensity = spectrum[y, x]
        
        # Calculate local area properties
        region = spectrum[max(0, y-2):min(spectrum.shape[0], y+3),
                         max(0, x-2):min(spectrum.shape[1], x+3)]
        
        properties = {
            'y_position': y,
            'x_position': x,
            'intensity': intensity,
            'local_max': np.max(region),
            'local_mean': np.mean(region),
            'local_std': np.std(region),
            'local_sum': np.sum(region)
        }
        peak_properties.append(properties)
    
    return pd.DataFrame(peak_properties)

def extract_regions_of_interest(spectrum, peak_positions, region_size=5):
    """
    Extract spectral regions around identified peaks
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    peak_positions: Array of (y, x) peak positions
    region_size: Size of region to extract around each peak
    
    Returns:
    List of extracted regions and their properties
    """
    regions = []
    for y, x in peak_positions:
        # Define region boundaries
        y_start = max(0, y - region_size)
        y_end = min(spectrum.shape[0], y + region_size + 1)
        x_start = max(0, x - region_size)
        x_end = min(spectrum.shape[1], x + region_size + 1)
        
        # Extract region
        region = spectrum[y_start:y_end, x_start:x_end]
        
        # Calculate region properties
        properties = {
            'center_y': y,
            'center_x': x,
            'mean_intensity': np.mean(region),
            'max_intensity': np.max(region),
            'min_intensity': np.min(region),
            'std_intensity': np.std(region),
            'sum_intensity': np.sum(region),
            'skewness': skew(region.flatten()),
            'kurtosis': kurtosis(region.flatten()),
            'area': region.size,
            'region_data': region
        }
        regions.append(properties)
    
    return regions

def calculate_peak_volumes(spectrum, peak_df, integration_radius=3):
    """
    Calculate volumes of peaks using integration
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    peak_df: DataFrame with peak positions
    integration_radius: Radius for volume integration
    
    Returns:
    DataFrame with added volume information
    """
    volumes = []
    for _, peak in peak_df.iterrows():
        y, x = int(peak['y_position']), int(peak['x_position'])
        
        # Define integration region
        y_start = max(0, y - integration_radius)
        y_end = min(spectrum.shape[0], y + integration_radius + 1)
        x_start = max(0, x - integration_radius)
        x_end = min(spectrum.shape[1], x + integration_radius + 1)
        
        # Calculate volume
        region = spectrum[y_start:y_end, x_start:x_end]
        volume = np.sum(region)
        volumes.append(volume)
    
    peak_df['volume'] = volumes
    return peak_df

def extract_connectivity_features(spectrum, peaks_df, threshold=0.1):
    """
    Extract connectivity patterns between peaks
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    peaks_df: DataFrame with peak information
    threshold: Minimum intensity threshold for connectivity
    
    Returns:
    Dictionary of connectivity features
    """
    peaks = peaks_df[['y_position', 'x_position']].values
    n_peaks = len(peaks)
    connectivity_matrix = np.zeros((n_peaks, n_peaks))
    
    for i in range(n_peaks):
        for j in range(i+1, n_peaks):
            # Get points along line between peaks
            y1, x1 = peaks[i]
            y2, x2 = peaks[j]
            points = list(zip(
                np.linspace(y1, y2, num=20).astype(int),
                np.linspace(x1, x2, num=20).astype(int)
            ))
            
            # Check intensity along connection
            intensities = [spectrum[y, x] for y, x in points]
            min_intensity = min(intensities)
            
            # Store connectivity strength
            if min_intensity > threshold:
                connectivity_matrix[i, j] = min_intensity
                connectivity_matrix[j, i] = min_intensity
    
    return {
        'connectivity_matrix': connectivity_matrix,
        'total_connections': np.sum(connectivity_matrix > 0) / 2,
        'average_connectivity': np.mean(connectivity_matrix[connectivity_matrix > 0]),
        'max_connectivity': np.max(connectivity_matrix)
    }

def extract_spectral_features(spectrum, ppm_x=None, ppm_y=None):
    """
    Extract global spectral features
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    ppm_x: Chemical shift values for x-axis (optional)
    ppm_y: Chemical shift values for y-axis (optional)
    
    Returns:
    Dictionary of spectral features
    """
    features = {
        'total_intensity': np.sum(spectrum),
        'mean_intensity': np.mean(spectrum),
        'max_intensity': np.max(spectrum),
        'min_intensity': np.min(spectrum),
        'std_intensity': np.std(spectrum),
        'skewness': skew(spectrum.flatten()),
        'kurtosis': kurtosis(spectrum.flatten()),
        'non_zero_fraction': np.mean(spectrum > 0),
        'intensity_quartiles': np.percentile(spectrum, [25, 50, 75])
    }
    
    # Add chemical shift range if provided
    if ppm_x is not None and ppm_y is not None:
        features.update({
            'x_shift_range': np.ptp(ppm_x),
            'y_shift_range': np.ptp(ppm_y)
        })
    
    return features

def extract_all_features(spectrum, ppm_x=None, ppm_y=None, 
                        peak_params=None, connectivity_threshold=0.1):
    """
    Complete feature extraction pipeline
    
    Parameters:
    spectrum: 2D numpy array of spectral data
    ppm_x: Chemical shift values for x-axis (optional)
    ppm_y: Chemical shift values for y-axis (optional)
    peak_params: Dictionary of peak detection parameters
    connectivity_threshold: Threshold for connectivity analysis
    
    Returns:
    Dictionary containing all extracted features
    """
    if peak_params is None:
        peak_params = {
            'height_threshold': 0.1,
            'min_distance': 5,
            'prominence': 0.05
        }
    
    # Find peaks
    peaks_df = find_peaks_2d(spectrum, **peak_params)
    
    # Calculate peak volumes
    peaks_df = calculate_peak_volumes(spectrum, peaks_df)
    
    # Extract regions of interest
    roi_features = extract_regions_of_interest(
        spectrum, 
        peaks_df[['y_position', 'x_position']].values
    )
    
    # Get connectivity features
    connectivity = extract_connectivity_features(
        spectrum, 
        peaks_df, 
        threshold=connectivity_threshold
    )
    
    # Get global spectral features
    spectral_features = extract_spectral_features(spectrum, ppm_x, ppm_y)
    
    return {
        'peaks': peaks_df,
        'regions_of_interest': roi_features,
        'connectivity': connectivity,
        'spectral_features': spectral_features
    }

# Example usage
if __name__ == "__main__":
    # Load preprocessed spectrum
    spectrum = np.load("preprocessed_spectrum.npy")
    
    # Extract all features
    features = extract_all_features(
        spectrum,
        peak_params={
            'height_threshold': 0.1,
            'min_distance': 5,
            'prominence': 0.05
        },
        connectivity_threshold=0.1
    )