import numpy as np
import os
import json
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import argparse
import pickle
import pendulum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dynamic_timeseries_dashboard_new import get_telemetry_data, ElasticsearchCloudQuery


class BaselineManager:
    """Manages scale factors for anomaly detection"""
    
    def __init__(self):
        self.deformation_scale = 0.0
        self.scale_factors_file = 'rw_anomaly_detection/models/scale_factors.json'
        
    def update_max_baseline(self, baseline_series):
        """No-op for max baseline update"""
        pass
        
    def compute_deformation_scale(self, baseline_series):
        """Compute deformation scale based on typical variations"""
        diffs = np.diff(baseline_series)
        typical_variation = np.nanstd(diffs)
        self.deformation_scale = typical_variation
        
    def save_factors(self):
        """Save scale factors to file"""
        os.makedirs(os.path.dirname(self.scale_factors_file), exist_ok=True)
        factors = {
            'deformation_scale': float(self.deformation_scale)
        }
        with open(self.scale_factors_file, 'w') as f:
            json.dump(factors, f)
            
    def load_factors(self):
        """Load scale factors from file"""
        if not os.path.exists(self.scale_factors_file):
            raise ValueError("No saved scale factors found. Must run training first.")
        with open(self.scale_factors_file, 'r') as f:
            factors = json.load(f)
        self.deformation_scale = factors['deformation_scale']
        
    def normalize_baseline(self, baseline_series):
        """No-op for baseline normalization, returns input as is"""
        return baseline_series
    
# Create necessary directories
os.makedirs('rw_anomaly_detection/models', exist_ok=True)
os.makedirs('rw_anomaly_detection/plots', exist_ok=True)

# Device-specific feature mapping with just the 4 required features
DEVICE_FEATURES = {
    1: {
        "DeformedCurrent": "deformed_current_rw1",
        "NormalizedCurrent": "normalized_current_rw1",
        "ResidualSpeed": "residual_speed_error_rw1",
        "Temperature": "spacecraft_adcs_measHealth_rwTemps_1"
    },
    2: {
        "DeformedCurrent": "deformed_current_rw2",
        "NormalizedCurrent": "normalized_current_rw2",
        "ResidualSpeed": "residual_speed_error_rw2",
        "Temperature": "spacecraft_adcs_measHealth_rwTemps_2"
    },
    3: {
        "DeformedCurrent": "deformed_current_rw3",
        "NormalizedCurrent": "normalized_current_rw3",
        "ResidualSpeed": "residual_speed_error_rw3",
        "Temperature": "spacecraft_adcs_measHealth_rwTemps_3"
    },
    4: {
        "DeformedCurrent": "deformed_current_rw4",
        "NormalizedCurrent": "normalized_current_rw4",
        "ResidualSpeed": "residual_speed_error_rw4",
        "Temperature": "spacecraft_adcs_measHealth_rwTemps_4"
    }
}

# Feature types for organization and filtering
FEATURE_TYPES = {
    "DeformedCurrent": [f"deformed_current_rw{i}" for i in range(1, 5)],
    "NormalizedCurrent": [f"normalized_current_rw{i}" for i in range(1, 5)],
    "ResidualSpeed": [f"residual_speed_error_rw{i}" for i in range(1, 5)],
    "Temperature": [f"spacecraft_adcs_measHealth_rwTemps_{i}" for i in range(1, 5)]
}

# Default acceptable thresholds for each feature type
DEFAULT_FEATURE_THRESHOLDS = {
    "Temperature": 4,  
    "DeformedCurrent": 8,  
    "NormalizedCurrent": 30,  
    "ResidualSpeed": 11
}

# Required raw telemetry points for feature computation
RAW_TELEMETRY_POINTS = {
    1: [
        "spacecraft_adcs_measHealth_rwCurrents_1",
        "spacecraft_adcs_measHealth_rwTemps_1",
        "spacecraft_adcs_measHealth_rwTargetSpeed_1",
        "spacecraft_adcs_meas_rw1Speed",
        "spacecraft_adcs_afswOut_acMode"
    ],
    2: [
        "spacecraft_adcs_measHealth_rwCurrents_2",
        "spacecraft_adcs_measHealth_rwTemps_2",
        "spacecraft_adcs_measHealth_rwTargetSpeed_2",
        "spacecraft_adcs_meas_rw2Speed",
        "spacecraft_adcs_afswOut_acMode"
    ],
    3: [
        "spacecraft_adcs_measHealth_rwCurrents_3",
        "spacecraft_adcs_measHealth_rwTemps_3",
        "spacecraft_adcs_measHealth_rwTargetSpeed_3",
        "spacecraft_adcs_meas_rw3Speed",
        "spacecraft_adcs_afswOut_acMode"
    ],
    4: [
        "spacecraft_adcs_measHealth_rwCurrents_4",
        "spacecraft_adcs_measHealth_rwTemps_4",
        "spacecraft_adcs_measHealth_rwTargetSpeed_4",
        "spacecraft_adcs_meas_rw4Speed",
        "spacecraft_adcs_afswOut_acMode"
    ]
}

# Device colors for plotting
DEVICE_COLORS = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}

# Savitzky-Golay filter parameters
SAVGOL_PARAMS = {
    "DeformedCurrent": {"window_length": 61, "polyorder": 1},
    "NormalizedCurrent": {"window_length": 61, "polyorder": 1},
    "ResidualSpeed": {"window_length": 61, "polyorder": 1},
    "Temperature": {"window_length": 61, "polyorder": 1}
}

# Feature limits
DEFAULT_FEATURE_LIMITS = {
    "Temperature": {"lower": -10, "upper": 120},
    "DeformedCurrent": {"lower": -np.inf, "upper": np.inf},
    "NormalizedCurrent": {"lower": -np.inf, "upper": np.inf},
    "ResidualSpeed": {"lower": -np.inf, "upper": np.inf}
}

def get_device_telemetry_points(selected_devices):
    """Get raw telemetry points needed for selected devices"""
    telemetry_points = []
    for device_num in selected_devices:
        if device_num in RAW_TELEMETRY_POINTS:
            telemetry_points.extend(RAW_TELEMETRY_POINTS[device_num])
    return list(set(telemetry_points))  # Remove duplicates

def compute_mode_based_baseline(df, device_num, mode_col, min_segment_length=30):
    """Computes baseline by tracking mode 5 segments, smoothing brief transitions"""
    current_col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
    
    # Create mode segments
    mode_5 = df[mode_col] == 5
    mode_changes = mode_5.ne(mode_5.shift()).cumsum()
    
    # Initialize baseline series
    baseline = pd.Series(index=df.index, dtype=float)
    
    # First pass: compute baseline for all segments
    segment_stats = {}
    for segment_id in mode_changes.unique():
        segment_mask = mode_changes == segment_id
        segment_data = df.loc[segment_mask, current_col]
        
        if not segment_data.empty:
            segment_mean = segment_data.mean()
            segment_length = len(segment_data)
            baseline.loc[segment_mask] = segment_mean
            
            # Store segment statistics
            segment_stats[segment_id] = {
                'mean': segment_mean,
                'length': segment_length,
                'start_idx': segment_data.index[0],
                'end_idx': segment_data.index[-1]
            }
    
    # Second pass: handle brief segments
    for segment_id, stats in segment_stats.items():
        if stats['length'] < min_segment_length:
            # Find nearest longer segments
            prior_segments = {
                sid: s for sid, s in segment_stats.items()
                if s['end_idx'] < stats['start_idx'] and s['length'] >= min_segment_length
            }
            
            if prior_segments:
                # Use the most recent stable segment's mean
                latest_prior = max(prior_segments.items(), key=lambda x: x[1]['end_idx'])
                segment_mask = mode_changes == segment_id
                baseline.loc[segment_mask] = latest_prior[1]['mean']
    
    return baseline

def create_anomalous_deformation(baseline, ac_mode, deform_factor=0.5, min_threshold=0.01, 
                               use_noise=False, noise_factor=0.5, baseline_manager=None, 
                               training_mode=False):
    """Creates deformations based on mode-specific baseline"""
    if baseline_manager is None:
        raise ValueError("BaselineManager instance required")
    
    if training_mode:
        baseline_manager.update_max_baseline(baseline)
        baseline_manager.compute_deformation_scale(baseline)
        normalized_baseline = baseline - baseline.iloc[0]
    else:
        normalized_baseline = baseline_manager.normalize_baseline(baseline - baseline.iloc[0])
    
    abs_baseline = np.abs(normalized_baseline)
    max_abs = np.nanmax(abs_baseline)
    normalized_magnitude = abs_baseline / max_abs if max_abs != 0 else np.zeros_like(abs_baseline)
    
    abs_baseline_np = abs_baseline.to_numpy()
    normalized_magnitude_np = normalized_magnitude.to_numpy()
    baseline_np = normalized_baseline.to_numpy()
    ac_mode_np = ac_mode.to_numpy()
    
    if use_noise:
        noise = np.zeros_like(baseline_np)
        mask = abs_baseline_np > min_threshold
        baseline_std = np.nanstd(abs_baseline_np)
        base_noise_scale = baseline_std * noise_factor
        
        for idx in range(len(baseline_np)):
            if mask[idx]:
                mode_factor = 1.2 if ac_mode_np[idx] == 1 else 0.8
                relative_magnitude = abs_baseline_np[idx] / max_abs
                local_scale = base_noise_scale * relative_magnitude * mode_factor
                noise[idx] = np.random.normal(0, local_scale)
        
        return pd.Series(baseline_np + noise, index=baseline.index)
    else:
        deformation = np.zeros_like(baseline_np)
        mask = abs_baseline_np > min_threshold
        
        for idx in range(len(baseline_np)):
            if mask[idx]:
                magnitude = normalized_magnitude_np[idx]
                mode_factor = 1.2 if ac_mode_np[idx] == 1 else 0.8
                base_scale = baseline_manager.deformation_scale
                
                shift = np.sin(idx/20) * magnitude * base_scale * mode_factor
                spike = (np.random.choice([-1, 1]) if np.random.random() < magnitude else 0) * magnitude * base_scale
                step = (0.5 if np.random.random() < magnitude/2 else -0.2) * magnitude * base_scale
                
                deformation[idx] = (shift + spike + step) * deform_factor
        
        return pd.Series(baseline_np + deformation, index=baseline.index)
    
def compute_normalized_current(df, device_num, min_speed_threshold=0.0001):
    """Computes normalized current with minimum speed threshold"""
    current_col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
    speed_col = f'spacecraft_adcs_meas_rw{device_num}Speed'
    
    # Create mask for acceptable speeds
    speed_mask = abs(df[speed_col]) >= min_speed_threshold
    
    # Initialize normalized current series
    normalized = pd.Series(0, index=df.index)
    
    # Compute normalized current only where speed is above threshold
    normalized[speed_mask] = df.loc[speed_mask, current_col] / df.loc[speed_mask, speed_col].abs()
    
    return normalized

def compute_residual_speed(df, device_num):
    """Compute residual speed error for a specific device."""
    target_speed_col = f'spacecraft_adcs_measHealth_rwTargetSpeed_{device_num}'
    speed_col = f'spacecraft_adcs_meas_rw{device_num}Speed'
    return df[target_speed_col] - df[speed_col]

def compute_device_features(df, device_num, deform_factor=0.5, min_threshold=0.1, 
                          baseline_manager=None, training_mode=False):
    """Computes device features using mode-based baseline"""
    result_df = df.copy()
    mode_col = 'spacecraft_adcs_afswOut_acMode'
    
    # Compute baseline and save it with device-specific name
    baseline = compute_mode_based_baseline(
        result_df,
        device_num,
        mode_col
    )
    result_df[f'baseline_rw{device_num}'] = baseline
    
    # Add deformed current
    result_df[f'deformed_current_rw{device_num}'] = create_anomalous_deformation(
        baseline,
        result_df[mode_col],
        deform_factor,
        min_threshold,
        baseline_manager=baseline_manager,
        training_mode=training_mode
    )
    
    # Add normalized current
    result_df[f'normalized_current_rw{device_num}'] = compute_normalized_current(
        result_df, device_num
    )
    
    # Add residual speed error
    result_df[f'residual_speed_error_rw{device_num}'] = compute_residual_speed(
        result_df, device_num
    )
    
    return result_df

def compute_all_features(df, selected_devices, deform_factor=0.5, min_threshold=0.1,
                        baseline_manager=None, training_mode=False):
    """Compute all derived features for selected devices."""
    print("\nComputing derived features...")
    result_df = df.copy()
    
    if 'spacecraft_adcs_afswOut_acMode' not in df.columns:
        raise ValueError("Required column 'spacecraft_adcs_afswOut_acMode' not found in dataframe")
    
    for device_num in selected_devices:
        print(f"\nProcessing Device {device_num}")
        try:
            required_cols = [
                f'spacecraft_adcs_measHealth_rwCurrents_{device_num}',
                f'spacecraft_adcs_meas_rw{device_num}Speed',
                f'spacecraft_adcs_measHealth_rwTargetSpeed_{device_num}',
                'spacecraft_adcs_afswOut_acMode'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"WARNING: Missing required columns for device {device_num}: {missing_cols}")
                continue
            
            result_df = compute_device_features(
                result_df, device_num,
                deform_factor, min_threshold,
                baseline_manager=baseline_manager,
                training_mode=training_mode
            )
            
        except Exception as e:
            print(f"Error computing features for device {device_num}: {str(e)}")
            continue
    
    return result_df

def apply_filters(df, feature_types=None):
    """Apply Savitzky-Golay filters to specified feature types."""
    if feature_types is None:
        feature_types = SAVGOL_PARAMS.keys()
    
    filtered_df = df.copy()
    for feature_type in feature_types:
        if feature_type in SAVGOL_PARAMS:
            params = SAVGOL_PARAMS[feature_type]
            feature_columns = []
            
            # Get all columns that match this feature type
            for pattern in FEATURE_TYPES[feature_type]:
                feature_columns.extend([col for col in filtered_df.columns if pattern in col])
            
            # Apply filter to matching columns
            for column in feature_columns:
                if column in filtered_df.columns:
                    filtered_df[column] = savgol_filter(
                        filtered_df[column],
                        window_length=params["window_length"],
                        polyorder=params["polyorder"]
                    )
    
    return filtered_df

def apply_feature_limits(df, use_limits=False):
    """Apply upper and lower limits to features and remove out-of-bounds values."""
    if not use_limits:
        return df
    
    mask = pd.Series(True, index=df.index)
    removed_counts = {}
    
    for feature_type, limits in DEFAULT_FEATURE_LIMITS.items():
        feature_columns = []
        for pattern in FEATURE_TYPES[feature_type]:
            feature_columns.extend([col for col in df.columns if pattern in col])
            
        for column in feature_columns:
            if column in df.columns:
                feature_mask = (df[column] >= limits["lower"]) & (df[column] <= limits["upper"])
                removed_counts[column] = (~feature_mask).sum()
                mask &= feature_mask
    
    filtered_df = df[mask].copy()
    
    if len(df) - len(filtered_df) > 0:
        print("\nRows removed due to limit violations:")
        for feature, count in removed_counts.items():
            if count > 0:
                print(f"{feature}: {count} rows")
        print(f"Total rows removed: {len(df) - len(filtered_df)}")
    
    return filtered_df

def compute_feature_bounds(data_standardized, device_num):
    """Computes max absolute values for each feature"""
    bounds = {}
    for feature_type, feature_col in DEVICE_FEATURES[device_num].items():
        if feature_col in data_standardized.columns:
            bounds[feature_type] = float(abs(data_standardized[feature_col]).max())
            print(f"Computing bound for {feature_type}: {bounds[feature_type]}")
    
    return bounds

def save_feature_bounds(device_num, bounds):
    """Saves feature bounds for a device"""
    filename = f'rw_anomaly_detection/models/device_{device_num}_feature_bounds.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(bounds, f)

def load_feature_bounds(device_num):
    """Loads feature bounds for a device"""
    filename = f'rw_anomaly_detection/models/device_{device_num}_feature_bounds.json'
    if not os.path.exists(filename):
        raise ValueError(f"No feature bounds found for device {device_num}")
    with open(filename, 'r') as f:
        return json.load(f)

def apply_feature_bounds(data, device_num, thresholds=None):
    """Applies feature bounds and threshold-based filtering"""
    result = data.copy()
    bounds = load_feature_bounds(device_num)
    thresholds = thresholds or DEFAULT_FEATURE_THRESHOLDS
    
    print(f"\nApplying bounds and thresholds for device {device_num}:")
    for feature_type, bound in bounds.items():
        feature_col = DEVICE_FEATURES[device_num][feature_type]
        if feature_col in result.columns:
            # Get absolute values for comparison
            abs_values = abs(result[feature_col])
            original_signs = np.sign(result[feature_col])
            
            # Create mask for within bounds
            within_bounds = abs_values <= bound
            exceeds_bounds = ~within_bounds
            
            # Get stats before modification
            max_val = abs_values.max()
            print(f"{feature_type}:")
            print(f"  Bound: {bound}")
            print(f"  Threshold: {thresholds[feature_type]}")
            print(f"  Max absolute value: {max_val}")
            
            # Apply standard bounding logic
            result.loc[within_bounds, feature_col] = 0
            
            if exceeds_bounds.any():
                exceedance = abs_values[exceeds_bounds] - bound
                max_exceedance = exceedance.max()
                
                # Check if max bounded value exceeds threshold
                if max_exceedance <= thresholds[feature_type]:
                    print(f"  Max exceedance ({max_exceedance}) below threshold, zeroing feature")
                    result[feature_col] = 0
                else:
                    # Apply normal bounding logic with time-based variations
                    normalized_exceedance = exceedance / max_exceedance if max_exceedance > 0 else exceedance
                    time_indices = np.arange(len(exceedance))
                    
                    # Generate rich noise profile
                    base_freq = 0.1
                    noise_components = []
                    freqs = [1, 2, 3]
                    amplitudes = [1.0, 0.5, 0.25]
                    
                    for freq, amp in zip(freqs, amplitudes):
                        wave = np.sin(time_indices * base_freq * freq) * normalized_exceedance * amp
                        noise_components.append(wave)
                    
                    noise_scale = 0.3
                    combined_noise = sum(noise_components) * noise_scale
                    
                    random_scale = 0.1
                    random_noise = np.random.normal(0, 1, len(exceedance)) * normalized_exceedance * random_scale
                    
                    total_variation = (combined_noise + random_noise) * exceedance
                    result.loc[exceeds_bounds, feature_col] = total_variation * original_signs[exceeds_bounds]
            
            # Log results
            num_zeroed = (result[feature_col] == 0).sum()
            num_total = len(result[feature_col])
            print(f"  Final zero count: {num_zeroed}/{num_total} points ({num_zeroed/num_total*100:.2f}%)")
            
            if (result[feature_col] != 0).any():
                remaining_max = abs(result[feature_col]).max()
                print(f"  Max absolute value after processing: {remaining_max}")
    
    return result

def validate_device_features():
    """Validate device features configuration"""
    print("\nValidating device features configuration...")
    
    # Check for consistency in feature names across devices
    all_feature_names = set()
    for device_num, features in DEVICE_FEATURES.items():
        all_feature_names.update(features.keys())
    
    print(f"Found feature types: {sorted(all_feature_names)}")
    
    # Validate each device has all features
    for device_num, features in DEVICE_FEATURES.items():
        missing_features = all_feature_names - set(features.keys())
        if missing_features:
            print(f"WARNING: Device {device_num} is missing features: {missing_features}")
    
    # Validate FEATURE_TYPES matches DEVICE_FEATURES
    feature_type_names = set(FEATURE_TYPES.keys())
    if feature_type_names != all_feature_names:
        print("WARNING: Mismatch between FEATURE_TYPES and DEVICE_FEATURES")
        print(f"Features in DEVICE_FEATURES: {all_feature_names}")
        print(f"Features in FEATURE_TYPES: {feature_type_names}")
        extra_in_device = all_feature_names - feature_type_names
        extra_in_types = feature_type_names - all_feature_names
        if extra_in_device:
            print(f"Extra in DEVICE_FEATURES: {extra_in_device}")
        if extra_in_types:
            print(f"Extra in FEATURE_TYPES: {extra_in_types}")
            
def load_telemetry_data(selected_devices, use_filter=True, use_limits=True, deform_factor=0.5, min_threshold=0.1):
    """Load and preprocess telemetry data."""
    telemetry_puller = ElasticsearchCloudQuery(context='telemetry', environment='prod')
    
    # Initialize baseline manager
    baseline_manager = BaselineManager()
    
    # Get telemetry points only for selected devices
    print("\nGetting telemetry points for selected devices...")
    telemetry_points = get_device_telemetry_points(selected_devices)
    print(f"Selected telemetry points: {telemetry_points}")
    
    if not telemetry_points:
        raise ValueError("No valid telemetry points found for selected devices") # '2021-12-20T00:14:15', '2021-12-27T02:14:16' #
    
    start_time, end_time =  '2024-12-18T14:29:13', '2024-12-18T20:29:52' # '2021-12-16T00:14:15', '2021-12-16T02:14:16' # 
    print(f"\nFetching data from {start_time} to {end_time}")
    start = pendulum.parse(start_time)
    end = pendulum.parse(end_time)
    spacecraft_id = '114'
    
    df = get_telemetry_data(telemetry_puller, spacecraft_id, start, end, 
                           telemetry_points, debug=True, bypass_cache=False)
    
    print(f"\nInitial dataframe shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Compute derived features only for selected devices
    print("\nComputing derived features...")
    df = compute_all_features(df, selected_devices, deform_factor, min_threshold,
                            baseline_manager=baseline_manager, training_mode=True)
    print(f"Dataframe shape after feature computation: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    if use_limits:
        print("\nApplying feature limits...")
        df = apply_feature_limits(df, use_limits=True)
        print(f"Dataframe shape after limits: {df.shape}")
    
    if use_filter:
        print("\nApplying Savitzky-Golay filtering...")
        filtered_df = apply_filters(df)
        print(f"Filtered dataframe shape: {filtered_df.shape}")
        return df, filtered_df, baseline_manager
    
    print("\nSkipping filtering as per configuration")
    return df, df, baseline_manager

def is_feature_blacklisted(feature, blacklisted_features):
    """Check if a feature should be blacklisted based on its type."""
    if not blacklisted_features:
        return False
        
    for feature_type in blacklisted_features:
        if feature_type not in FEATURE_TYPES:
            print(f"Warning: Unknown feature type '{feature_type}' in blacklist")
            continue
            
        if any(pattern in feature for pattern in FEATURE_TYPES[feature_type]):
            return True
            
    return False

def get_device_features(device_num, blacklisted_features=None):
    """Get device features excluding blacklisted ones."""
    all_features = DEVICE_FEATURES[device_num]
    
    if not blacklisted_features:
        return all_features
        
    return {
        feature_name: feature_col
        for feature_name, feature_col in all_features.items()
        if not is_feature_blacklisted(feature_col, blacklisted_features)
    }

def process_device_data(df, filtered_df, device_num, scaler=None, pca=None, 
                       filter_standardized=False, blacklisted_features=None, 
                       skip_pca=False, baseline_manager=None, training_mode=False):
    """Process data for a single device."""
    print(f"\nProcessing data for Device {device_num}")
    
    device_features = get_device_features(device_num, blacklisted_features)
    print(f"Selected features: {device_features}")
    
    # Store intermediate features before standardization
    intermediate_data = {
        'raw_features': filtered_df[list(device_features.values())].copy()
    }
    
    # Get baseline current information if available
    current_feature = f'deformed_current_rw{device_num}'
    baseline_feature = f'baseline_rw{device_num}'
    if current_feature in filtered_df.columns and baseline_feature in df.columns:
        intermediate_data.update({
            'baseline_current': df[baseline_feature].copy(),
            'normalized_baseline': df[baseline_feature] - df[baseline_feature].iloc[0],
            'deformed_current': filtered_df[current_feature].copy()
        })
    
    feature_list = list(device_features.values())
    print(f"Feature list for processing: {feature_list}")
    
    device_df = df[feature_list] if filter_standardized else filtered_df[feature_list]
    
    # Standardization
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(device_df)
    else:
        data_scaled = scaler.transform(device_df)
    
    data_standardized = pd.DataFrame(data_scaled, columns=device_df.columns, index=df.index)
    
    # When training, compute bounds after standardization but before filtering
    if training_mode:
        bounds = compute_feature_bounds(data_standardized, device_num)
        save_feature_bounds(device_num, bounds)
        print(f"Saved feature bounds during training: {bounds}")
    
    if filter_standardized:
        filtered_standardized = apply_filters(data_standardized)
        data_for_pca = filtered_standardized
    else:
        data_for_pca = data_standardized
    
    if skip_pca:
        pca = None
        pca_df = data_for_pca.copy()
        pca_df.columns = [f"Device{device_num}_Feature_{i+1}" for i in range(len(device_features))]
    else:
        if pca is None:
            pca = PCA()
            pca_results = pca.fit_transform(data_for_pca)
        else:
            pca_results = pca.transform(data_for_pca)
        
        pca_columns = [f"Device{device_num}_Component_{i+1}" for i in range(len(device_features))]
        pca_df = pd.DataFrame(pca_results, columns=pca_columns, index=df.index)
    
    return data_standardized, pca_df, scaler, pca, intermediate_data

def validate_feature_order(device_num, feature_list, blacklisted_features=None):
    """Validate that features are in the expected order, considering blacklisted features."""
    device_features = get_device_features(device_num, blacklisted_features)
    expected_features = list(device_features.values())
    
    if feature_list != expected_features:
        raise ValueError(f"Feature order mismatch for device {device_num}. "
                        f"Expected: {expected_features}, Got: {feature_list}")

def process_data_for_prediction(df, filtered_df, device_num, model_data, filter_standardized=False, blacklisted_features=None):
    """Process data for prediction using a trained model."""
    device_features = get_device_features(device_num, blacklisted_features)
    
    if not device_features:
        raise ValueError(f"No features remaining for device {device_num} after applying blacklist")
    
    feature_list = list(device_features.values())
    
    # Validate feature order with blacklist consideration
    print(f"\nValidating feature order for device {device_num}")
    print(f"Expected features: {feature_list}")
    validate_feature_order(device_num, feature_list, blacklisted_features)
    
    device_df = df[feature_list] if filter_standardized else filtered_df[feature_list]
    print(f"Device dataframe shape: {device_df.shape}")
    
    # Apply the same standardization from training
    data_scaled = model_data['scaler'].transform(device_df)
    data_standardized = pd.DataFrame(data_scaled, columns=device_df.columns, index=df.index)
    
    if filter_standardized:
        filtered_standardized = apply_filters(data_standardized)
        data_for_pca = filtered_standardized
    else:
        data_for_pca = data_standardized
    
    # Check if model was trained with PCA
    if model_data.get('skip_pca', False):
        print("\nUsing standardized features directly (PCA was skipped in training)")
        pca_df = data_for_pca.copy()
        pca_df.columns = [f"Device{device_num}_Feature_{i+1}" for i in range(len(device_features))]
    else:
        # Apply the same PCA transformation from training
        pca_results = model_data['pca'].transform(data_for_pca)
        pca_columns = [f"Device{device_num}_Component_{i+1}" for i in range(len(device_features))]
        pca_df = pd.DataFrame(pca_results, columns=pca_columns, index=df.index)
    
    return data_standardized, pca_df

def predict(df, filtered_df, device_num, filter_standardized=False, blacklisted_features=None, baseline_manager=None):
    """Makes predictions using feature bounds and returns standardized, bounded features, and predictions."""
    print(f"\nLoading model for Device {device_num}")
    model_data = load_model(device_num)
    
    try:
        # Process data for prediction
        data_standardized, pca_df = process_data_for_prediction(
            df, filtered_df, device_num,
            model_data,
            filter_standardized=filter_standardized,
            blacklisted_features=blacklisted_features
        )
        
        # Apply feature bounds to standardized data
        print("\nApplying feature bounds to standardized data...")
        data_bounded = apply_feature_bounds(data_standardized, device_num)
        
        # Transform bounded data to PCA space
        if not model_data.get('skip_pca', False):
            print("\nApplying PCA transformation...")
            pca_results = model_data['pca'].transform(data_bounded)
            pca_columns = [f"Device{device_num}_Component_{i+1}" 
                         for i in range(len(get_device_features(device_num, blacklisted_features)))]
            data_for_prediction = pd.DataFrame(pca_results, columns=pca_columns, index=df.index)
        else:
            data_for_prediction = data_bounded
            
        # Drop last 3 minutes of data before prediction
        cutoff_time = data_for_prediction.index[-1] - pd.Timedelta(minutes=3)
        data_for_prediction = data_for_prediction[data_for_prediction.index <= cutoff_time]
        print(f"\nDropped last 3 minutes of data. Rows for prediction: {len(data_for_prediction)}")
        
        print("\nMaking predictions...")
        # Get raw predictions array (-1 for inliers, 1 for outliers)
        raw_predictions = model_data['model'].predict(data_for_prediction)
        
        # Convert to binary array (1 for anomalies, 0 for normal)
        predictions = (raw_predictions == -1).astype(int)
        
        # Create full predictions array matching original data length
        full_predictions = pd.Series(index=df.index, dtype=int)
        full_predictions.loc[data_for_prediction.index] = predictions
        full_predictions.fillna(0, inplace=True)  # Fill dropped timestamps with 0 (non-anomalous)
        
        num_anomalies = np.sum(predictions == 1)
        print(f"Found {num_anomalies} anomalies ({(num_anomalies/len(predictions))*100:.2f}% of data points)")
        
        # Plot results including bounded features
        plot_device_anomalies(
            data_standardized=data_standardized, 
            pca_df=data_for_prediction,
            train_predictions=full_predictions,
            test_predictions=full_predictions,
            split_idx=len(data_standardized),
            device_num=device_num,
            blacklisted_features=blacklisted_features,
            data_bounded=data_bounded,
            prediction_mode=True
        )
        
        return full_predictions.copy()
        
    except Exception as e:
        print(f"Error in predict for device {device_num}: {str(e)}")
        raise

def train_model(df, filtered_df, device_num, contamination=0.01, filter_standardized=False, 
              blacklisted_features=None, skip_pca=False, baseline_manager=None, training_mode=True):
    """Train model with feature bound computation"""
    print(f"\nTraining model for Device {device_num}")
    
    try:
        # Process device data
        data_standardized, pca_df, scaler, pca, intermediate_data = process_device_data(
            df, filtered_df, device_num,
            filter_standardized=filter_standardized,
            blacklisted_features=blacklisted_features,
            skip_pca=skip_pca,
            baseline_manager=baseline_manager,
            training_mode=training_mode
        )
        
        # Save baseline manager's scale factors
        print("\nSaving baseline manager scale factors...")
        baseline_manager.save_factors()
        
        # Compute and save feature bounds
        print("\nComputing feature bounds...")
        bounds = compute_feature_bounds(data_standardized, device_num)
        save_feature_bounds(device_num, bounds)
        print(f"Saved bounds: {bounds}")
        
        # Apply bounds before training
        data_for_training = apply_feature_bounds(pca_df, device_num)
        
        split_idx = int(len(data_for_training) * 0.7)
        X_train = data_for_training.iloc[:split_idx]
        X_test = data_for_training.iloc[split_idx:]
        
        print("\nTraining Isolation Forest...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        train_predictions = (iso_forest.fit_predict(X_train) == -1).astype(int)
        test_predictions = (iso_forest.predict(X_test) == -1).astype(int)
        
        print("\nSaving model...")
        model_data = {
            'model': iso_forest,
            'scaler': scaler,
            'pca': pca,
            'filter_standardized': filter_standardized,
            'skip_pca': skip_pca,
            'blacklisted_features': blacklisted_features,
        }
        save_model(device_num, model_data)
        
        return data_standardized, pca_df, train_predictions, test_predictions, split_idx, intermediate_data
        
    except Exception as e:
        print(f"Error in train_model for device {device_num}: {str(e)}")
        raise

def save_model(device_num, model_data):
    """Save model data for a specific device."""
    filename = f'rw_anomaly_detection/models/device_{device_num}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(device_num):
    """Load model data for a specific device."""
    filename = f'rw_anomaly_detection/models/device_{device_num}_model.pkl'
    if not os.path.exists(filename):
        raise ValueError(f"No model file found for device {device_num}")
    with open(filename, 'rb') as f:
        return pickle.load(f)

def parse_devices(devices_str):
    """Parse devices string into list of device numbers."""
    if devices_str.lower() == 'all':
        return list(DEVICE_FEATURES.keys())
    try:
        devices = [int(d.strip()) for d in devices_str.split(',')]
        invalid_devices = [d for d in devices if d not in DEVICE_FEATURES]
        if invalid_devices:
            raise ValueError(f"Invalid device numbers: {invalid_devices}")
        return devices
    except ValueError as e:
        if "Invalid device numbers" in str(e):
            raise
        raise ValueError("Devices must be 'all' or comma-separated numbers (e.g., '1,2,4')")
    
def plot_standardized_features(data_standardized, device_num):
    """Plot standardized features for a single device."""
    print("here5")
    num_features = len(data_standardized.columns)
    rows = (num_features + 2) // 3
    print("here3")
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    print("here4")
    axes = axes.flatten()
    
    for i, feature in enumerate(data_standardized.columns):
        sns.lineplot(data=data_standardized[feature], ax=axes[i], label=feature)
        axes[i].set_title(f"Device {device_num} - Standardized {feature}")
        axes[i].set_xlabel("Time Index")
        axes[i].set_ylabel("Standardized Value")
        axes[i].legend()
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'rw_anomaly_detection/plots/device{device_num}_standardized_features.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(data, device_num, title_suffix=""):
    """Plot correlation matrix for a single device."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Device {device_num} - {title_suffix}")
    plt.savefig(f'rw_anomaly_detection/plots/device{device_num}_correlation_matrix{title_suffix.replace(" ", "_").lower()}.png')
    plt.close()

def plot_pca_distributions(pca_df, device_num):
    """Plot PCA component distributions for a single device."""
    num_components = len(pca_df.columns)
    rows = (num_components + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    for i, column in enumerate(pca_df.columns):
        sns.histplot(pca_df[column], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f"Device {device_num} - {column} Distribution")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'rw_anomaly_detection/plots/device{device_num}_pca_distributions.png')
    plt.close()

def plot_device_anomalies(data_standardized, pca_df, train_predictions, test_predictions, 
                         split_idx, device_num, blacklisted_features=None, 
                         show_intermediate=False, intermediate_data=None,
                         data_bounded=None, prediction_mode=False):
    """Plots features and PCA components with anomalies, including bounded features in prediction mode."""
    
    print(f"\nPlotting device anomalies for Device {device_num}")
    
    device_features = get_device_features(device_num, blacklisted_features)
    
    # Calculate number of plots needed
    n_standard_plots = len(device_features) + len(pca_df.columns)
    if prediction_mode and data_bounded is not None:
        n_standard_plots += len(device_features)  # Add plots for bounded features
    
    n_intermediate_plots = 0
    if show_intermediate and intermediate_data is not None:
        n_intermediate_plots = len(device_features) + 1
    
    total_plots = n_standard_plots + n_intermediate_plots
    rows = (total_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot intermediate processing results if requested
    if show_intermediate and intermediate_data is not None:
        # Plot raw features and baseline comparison
        # [Previous intermediate plotting code remains unchanged]
        plot_idx = n_intermediate_plots
    
    # Handle both training and prediction modes
    if prediction_mode:
        original_train = pd.DataFrame()
        original_test = data_standardized
        pca_train = pd.DataFrame()
        pca_test = pca_df
        train_anomalies = []
        test_anomalies = original_test[test_predictions == 1].index
    else:
        original_train = data_standardized.iloc[:split_idx]
        original_test = data_standardized.iloc[split_idx:]
        pca_train = pca_df.iloc[:split_idx]
        pca_test = pca_df.iloc[split_idx:]
        train_anomalies = original_train[train_predictions == 1].index
        test_anomalies = original_test[test_predictions == 1].index
    
    # Plot standardized features
    for feature_name, feature_col in device_features.items():
        ax = axes[plot_idx]
        
        if not original_train.empty:
            ax.plot(original_train.index, original_train[feature_col], 
                   label="Training", alpha=0.6, color='blue')
            if len(train_anomalies) > 0:
                ax.scatter(train_anomalies, 
                          original_train.loc[train_anomalies, feature_col],
                          color="black", label="Training Anomalies", zorder=8)
        
        ax.plot(original_test.index, original_test[feature_col], 
                label="Testing/Prediction", alpha=0.6, color='green')
        
        if len(test_anomalies) > 0:
            ax.scatter(test_anomalies, 
                      original_test.loc[test_anomalies, feature_col],
                      color="red", label="Anomalies", zorder=8)
        
        ax.set_title(f"Device {device_num} - Standardized {feature_name}")
        ax.legend(loc='upper right')
        plot_idx += 1
    
    # Plot bounded features if in prediction mode
    if prediction_mode and data_bounded is not None:
        for feature_name, feature_col in device_features.items():
            ax = axes[plot_idx]
            
            ax.plot(data_bounded.index, data_bounded[feature_col], 
                    label="Bounded Feature", alpha=0.6, color='purple')
            
            if len(test_anomalies) > 0:
                ax.scatter(test_anomalies, 
                          data_bounded.loc[test_anomalies, feature_col],
                          color="red", label="Anomalies", zorder=8)
            
            ax.set_title(f"Device {device_num} - Bounded {feature_name}")
            ax.legend(loc='upper right')
            plot_idx += 1
    
    # Plot PCA components
    for i, col in enumerate(pca_df.columns):
        ax = axes[plot_idx]
        
        if not pca_train.empty:
            ax.plot(pca_train.index, pca_train[col], 
                   label="Training PCA", alpha=0.6, color='blue')
            if len(train_anomalies) > 0:
                ax.scatter(train_anomalies, 
                          pca_train.loc[train_anomalies, col],
                          color="black", label="Training Anomalies", zorder=8)
        
        ax.plot(pca_test.index, pca_test[col], 
                label="Testing/Prediction PCA", alpha=0.6, color='green')
        
        if len(test_anomalies) > 0:
            ax.scatter(test_anomalies, 
                      pca_test.loc[test_anomalies, col],
                      color="red", label="Anomalies", zorder=8)
        
        ax.set_title(f"Device {device_num} - {col}")
        ax.legend(loc='upper right')
        plot_idx += 1
    
    # Remove any unused subplots
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'rw_anomaly_detection/plots/device{device_num}_anomalies.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_confusion_matrix(y_test, y_pred, device_num):
    """Plot confusion matrix for model predictions."""
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    if len(unique_classes) > 1:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
        disp.plot(cmap="Blues")
        plt.title(f"Device {device_num} - Isolation Forest Confusion Matrix")
        plt.savefig(f'rw_anomaly_detection/plots/device{device_num}_confusion_matrix.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Warning: Only one class present in predictions for Device {device_num}. Skipping confusion matrix plot.")

def plot_devices_comparison(df, filtered_df, all_predictions, selected_devices, blacklisted_features=None):
    """Plot time series data for selected devices together with their anomalies."""
    feature_types = [ft for ft in FEATURE_TYPES.keys() 
                    if not (blacklisted_features and ft in blacklisted_features)]
    
    if not feature_types:
        print("No features to plot after applying blacklist")
        return
    
    print(f"\nPlotting device comparison for features: {feature_types}")
    
    fig, axes = plt.subplots(len(feature_types), 1, figsize=(15, 5 * len(feature_types)))
    if len(feature_types) == 1:
        axes = [axes]
    
    for i, feature_type in enumerate(feature_types):
        ax = axes[i]
        for device_num in selected_devices:
            device_features = get_device_features(device_num, blacklisted_features)
            if feature_type in device_features:
                feature_name = device_features[feature_type]
                ax.plot(df.index, df[feature_name], 
                       label=f"Device {device_num}", 
                       color=DEVICE_COLORS[device_num],
                       alpha=0.6)
                
                anomalies = df[all_predictions[device_num] == 1].index
                if len(anomalies) > 0:
                    ax.scatter(anomalies, df.loc[anomalies, feature_name],
                              color=DEVICE_COLORS[device_num],
                              label=f"Device {device_num} Anomalies",
                              zorder=8)
        
        ax.set_title(f"{feature_type} Comparison Across Devices")
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Saving device comparison plot...")
    plt.savefig('rw_anomaly_detection/plots/devices_comparison.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def plot_raw_telemetry_metrics(df, all_predictions, selected_devices, save_path='rw_anomaly_detection/plots', filter_high_current_anomalies=True):
    """Plot raw telemetry metrics (current, speed, temperature) with anomalies for all devices together."""
    print("\nPlotting combined raw telemetry metrics")
    
    try:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Raw Telemetry Metrics - All Devices', fontsize=16, y=0.95)
        
        metrics = ['Current', 'Speed', 'Temperature']
        
        for ax, metric in zip(axes, metrics):
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3)
            
            for device_num in selected_devices:
                if device_num not in all_predictions:
                    continue
                
                if metric == 'Current':
                    col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
                elif metric == 'Speed':
                    col = f'spacecraft_adcs_meas_rw{device_num}Speed'
                else:  # Temperature
                    col = f'spacecraft_adcs_measHealth_rwTemps_{device_num}'
                
                ax.plot(df.index, df[col], 
                       label=f'Device {device_num}', 
                       color=DEVICE_COLORS[device_num],
                       alpha=0.7,
                       linewidth=1)
                
                predictions = all_predictions[device_num]
                anomaly_mask = predictions == 1
                anomaly_indices = df.index[anomaly_mask]
                
                if filter_high_current_anomalies and metric == 'Current':
                    current_col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
                    anomaly_indices = [idx for idx in anomaly_indices 
                                     if abs(df.loc[idx, current_col]) <= 1]
                
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, 
                             df.loc[anomaly_indices, col],
                             color=DEVICE_COLORS[device_num],
                             label=f'Device {device_num} Anomalies',
                             zorder=5)
            
            ax.set_ylabel(metric)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        plot_path = f'{save_path}/raw_metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating plots for device {device_num}: {str(e)}")
        plt.close()
        
def plot_all_devices_raw_metrics(df, all_predictions, selected_devices, save_path='rw_anomaly_detection/plots'):
    """Plot raw telemetry metrics for all selected devices."""
    print("\nGenerating raw telemetry metric plots for all devices...")
    
    for device_num in selected_devices:
        if device_num in all_predictions:
            try:
                plot_raw_telemetry_metrics(
                    df, 
                    device_num, 
                    all_predictions[device_num],
                    save_path
                )
            except Exception as e:
                print(f"Error plotting raw metrics for device {device_num}: {str(e)}")

def plot_raw_telemetry_metrics_plotly(df, all_predictions, selected_devices, save_path='rw_anomaly_detection/plots', filter_high_current_anomalies=True):
    """Plot raw telemetry metrics with Plotly for interactive visualization"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    print("\nPlotting combined raw telemetry metrics with Plotly")
    
    # Create subplots with updated titles
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            '<b>RW Current</b>',
            '<b>RW Speed</b>',
            '<b>RW Temperature</b>'
        ),
        vertical_spacing=0.1
    )
    
    # Updated metrics with units
    metrics = [
        ('Current', 'Current (A)'),
        ('Speed', 'Speed (rad/s)'),
        ('Temperature', 'Temperature (°C)')
    ]
    
    for idx, (metric, ylabel) in enumerate(metrics, 1):
        for device_num in selected_devices:
            if device_num not in all_predictions:
                continue
                
            # Get appropriate column name based on metric
            if metric == 'Current':
                col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
            elif metric == 'Speed':
                col = f'spacecraft_adcs_meas_rw{device_num}Speed'
            else:  # Temperature
                col = f'spacecraft_adcs_measHealth_rwTemps_{device_num}'
            
            # Add line trace for the metric
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=f'RW{device_num}',  # Simplified legend names
                    line=dict(color=DEVICE_COLORS[device_num], width=1),
                    opacity=0.7,
                    showlegend=True if idx == 1 else False  # Show legend only for first subplot
                ),
                row=idx, col=1
            )
            
            # Process anomalies
            predictions = all_predictions[device_num]
            anomaly_mask = predictions == 1
            anomaly_indices = df.index[anomaly_mask]
            
            if filter_high_current_anomalies and metric == 'Current':
                current_col = f'spacecraft_adcs_measHealth_rwCurrents_{device_num}'
                anomaly_indices = [idx for idx in anomaly_indices 
                                 if abs(df.loc[idx, current_col]) <= 1]
            
            if len(anomaly_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_indices,
                        y=df.loc[anomaly_indices, col],
                        mode='markers',
                        name=f'RW{device_num} Anomalies',  # Simplified anomaly names
                        marker=dict(
                            color=DEVICE_COLORS[device_num],
                            size=8,
                            symbol='circle'
                        ),
                        showlegend=True if idx == 1 else False
                    ),
                    row=idx, col=1
                )
    
    # Calculate legend y position based on number of devices
    legend_y = -0.15 - (0.02 * len(selected_devices))
    
    # Update layout with specified configurations
    fig.update_layout(
        title={
            'text': 'Reaction Wheel Telemetry',  # Updated main title
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'family': "Palatino Linotype"}
        },
        font={'family': 'Palatino Linotype', 'size': 12},
        plot_bgcolor='rgba(140, 151, 175, 0.1)',
        paper_bgcolor='rgba(140, 151, 175, 0.08)',
        showlegend=True,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': legend_y,
            'xanchor': 'center',
            'x': 0.5,
            'bgcolor': 'rgba(255, 255, 255, 0.7)',
            'font': {'family': "Palatino Linotype"}
        },
        height=900,
        margin=dict(t=100, b=100)
    )
    
    # Update axes labels and fonts with units
    for i in range(1, 4):
        # Update y-axis with units
        fig.update_yaxes(
            title_text=metrics[i-1][1],  # Use the label with units
            row=i, col=1,
            title_font=dict(family="Palatino Linotype", size=14),
            tickfont=dict(family="Palatino Linotype", size=12),
            gridcolor='rgba(140, 151, 175, 0.2)',
            showgrid=True
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text='Time' if i == 3 else '',
            row=i, col=1,
            title_font=dict(family="Palatino Linotype", size=14),
            tickfont=dict(family="Palatino Linotype", size=12),
            gridcolor='rgba(140, 151, 175, 0.2)',
            showgrid=True
        )
    
    # Save as HTML for interactivity
    plot_path = f'{save_path}/raw_metrics_comparison_plotly.html'
    fig.write_html(plot_path)
    
    print(f"Interactive plot saved to: {plot_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolation Forest for RW Friction Detection")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run prediction")
    parser.add_argument("--contamination", type=float, default=0.01,
                       help="Contamination level for Isolation Forest")
    parser.add_argument("--no-filter", action="store_true",
                       help="Disable Savitzky-Golay filtering")
    parser.add_argument("--filter-standardized", action="store_true",
                       help="Apply filtering to standardized values")
    parser.add_argument("--devices", type=str, default="all",
                       help="Comma-separated list of device numbers or 'all'")
    parser.add_argument("--blacklist", type=str,
                       help="Comma-separated list of feature types to exclude")
    parser.add_argument("--deform-factor", type=float, default=0,
                       help="Factor for deformation in anomaly simulation")
    parser.add_argument("--min-threshold", type=float, default=0.01,
                       help="Minimum threshold for deformation")
    parser.add_argument("--use-noise", action="store_true",
                       help="Use noise instead of deformation")
    parser.add_argument("--noise-factor", type=float, default=0.3,
                       help="Factor for noise magnitude")
    parser.add_argument("--skip-pca", action="store_true",
                       help="Skip PCA and use standardized features directly")
    args = parser.parse_args()

    try:
        # Parse device selection and blacklist
        selected_devices = parse_devices(args.devices)
        blacklisted_features = args.blacklist.split(',') if args.blacklist else None
        
        if blacklisted_features:
            print(f"Blacklisted feature types: {blacklisted_features}")
        
        # Load and preprocess data - now returns baseline_manager
        df, filtered_df, baseline_manager = load_telemetry_data(
            selected_devices,
            use_filter=not args.no_filter,
            deform_factor=args.deform_factor,
            min_threshold=args.min_threshold
        )

        if args.train:
            all_predictions = {}
            for device_num in selected_devices:
                try:
                    print(f"\nProcessing Device {device_num}")
                    
                    # Train model and get predictions
                    data_standardized, pca_df, train_pred, test_pred, split_idx, intermediate_data = train_model(
                        df, filtered_df, device_num,
                        args.contamination,
                        args.filter_standardized,
                        blacklisted_features,
                        skip_pca=args.skip_pca,
                        baseline_manager=baseline_manager,
                        training_mode=True
                    )
                    
                    # Store predictions for comparison plot
                    predictions = np.concatenate([train_pred, test_pred])
                    all_predictions[device_num] = predictions
                    print(f"Total predictions for device {device_num}: {len(predictions)}")
                    
                    try:
                        # Generate analysis plots
                        print(f"\nGenerating analysis plots for Device {device_num}...")
                        plot_standardized_features(data_standardized, device_num)
                        plot_correlation_matrix(data_standardized, device_num, "Feature Correlation")
                
                        if not args.skip_pca:
                            plot_pca_distributions(pca_df, device_num)
                            plot_correlation_matrix(pca_df, device_num, "PCA Correlation")
                        
                        # Generate anomaly plots
                        print(f"Generating anomaly plots for Device {device_num}...")
                        plot_device_anomalies(
                            data_standardized, pca_df, 
                            train_pred, test_pred,
                            split_idx, device_num, 
                            blacklisted_features,
                            show_intermediate=True, 
                            intermediate_data=intermediate_data
                        )
                        plot_confusion_matrix(test_pred, test_pred, device_num)
                        
                        # Generate raw metrics plot
                        print(f"Generating raw metrics plot for Device {device_num}...")
                        plot_raw_telemetry_metrics(df, device_num, predictions)
                        
                        print(f"Plot generation complete for Device {device_num}")
                        
                    except Exception as plot_error:
                        print(f"Error generating plots for device {device_num}: {str(plot_error)}")
                        continue
                    
                except Exception as e:
                    print(f"Error processing device {device_num}: {str(e)}")
                    continue
            
            # Plot device comparison if multiple devices
            if len(all_predictions) > 1:
                try:
                    print("\nGenerating comparison plots...")
                    print(f"Number of devices for comparison: {len(all_predictions)}")
                    print(f"Prediction lengths: {[len(pred) for pred in all_predictions.values()]}")
                    
                    # Generate device comparison plot
                    plot_devices_comparison(
                        df, filtered_df, all_predictions,
                        list(all_predictions.keys()),
                        blacklisted_features
                    )
                    
                    # Generate raw metrics comparison plot
                    plot_raw_telemetry_metrics(df, all_predictions, selected_devices)
                    
                    print("Comparison plots complete")
                except Exception as compare_error:
                    print(f"Error generating comparison plots: {str(compare_error)}")

        elif args.predict:
            all_predictions = {}  # Initialize empty dictionary
            baseline_manager = BaselineManager()
            baseline_manager.load_factors()

            for device_num in selected_devices:
                try:
                    print(f"\nPredicting for Device {device_num}")
                    # Store predictions in dictionary with device number as key
                    predictions = predict(
                        df, filtered_df, device_num,
                        args.filter_standardized,
                        blacklisted_features,
                        baseline_manager=baseline_manager
                    )
                    all_predictions[device_num] = predictions  # Store in dictionary
                    
                except Exception as e:
                    print(f"Error processing device {device_num}: {str(e)}")
                    continue
            
            # Plot device comparison if we have predictions
            if all_predictions:  # Check if dictionary is not empty
                try:
                    print("\nGenerating comparison plots...")
                    print(f"Number of devices for comparison: {len(all_predictions)}")
                    print(f"Prediction lengths: {[len(pred) for pred in all_predictions.values()]}")
                    
                    # Generate device comparison plot
                    plot_devices_comparison(
                        df, filtered_df, all_predictions,
                        list(all_predictions.keys()),
                        blacklisted_features
                    )
                    
                    # Generate raw metrics comparison plot
                    plot_raw_telemetry_metrics(df, all_predictions, selected_devices)
                    plot_raw_telemetry_metrics_plotly(df, all_predictions, selected_devices)
                    print("Comparison plots complete")
                except Exception as compare_error:
                    print(f"Error generating comparison plots: {str(compare_error)}")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        parser.print_help()
        exit(1)