#!/usr/bin/env python3
"""
Dynamite Fishing Detection System

This script implements a rule-based prototype for detecting dynamite fishing events 
in .wav files based on the hierarchical framework detailed in "Acoustic Signatures 
of Underwater Explosions: A Technical Report."

The detection system uses a three-tiered approach:
- Tier 1: Initial Event Detection (Candidate Identification)
- Tier 2: Primary Classification (Explosion Verification)  
- Tier 3: Contextual Filtering (Confuser Rejection)

"""

import os
import sys
import argparse
import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Configuration dictionary for all tunable thresholds
DETECTOR_CONFIG = {
    # Tier 1: Initial Event Detection
    "energy_threshold_db": 20,  # dB above ambient noise level
    "broadband_check_bands": [(100, 1000), (1000, 5000), (5000, 15000)],  # Hz frequency bands
    "broadband_energy_threshold": 0.15,  # Min energy percentage per band
    
    # Tier 2: Primary Classification  
    "shockwave_rise_time_ms": 5,  # Max rise time in milliseconds
    "bubble_pulse_search_window_ms": 500,  # Search window after shockwave (ms)
    "bubble_pulse_min_period_ms": 20,  # Min time between shockwave and bubble pulse
    "bubble_pulse_max_period_ms": 200,  # Max time between shockwave and bubble pulse
    "bubble_pulse_amplitude_ratio": 0.1,  # Min amplitude ratio (bubble/shockwave)
    "peak_prominence_factor": 0.3,  # For secondary peak detection
    
    # Tier 3: Contextual Filtering (for future use)
    "repetition_interval_s": 30,  # Time window for repetition checks
    
    # General parameters
    "sample_rate": 22050,  # Target sample rate for analysis
    "frame_length_ms": 50,  # Frame length for RMS analysis
    "min_event_duration_ms": 10,  # Minimum event duration
}


def analyze_audio_for_explosion(audio_path, config=DETECTOR_CONFIG):
    """
    Analyzes an audio file for dynamite explosion signatures based on the report.
    
    Args:
        audio_path (str): Path to the audio file
        config (dict): Configuration parameters
        
    Returns:
        dict: Analysis results with detection status and metrics
    """
    
    try:
        # Load audio file
        audio_data, sr = librosa.load(audio_path, sr=config["sample_rate"])
        
        if len(audio_data) == 0:
            return {
                'is_explosion': False,
                'reason': 'Empty audio file',
                'file_path': audio_path,
                'sample_rate': sr
            }
            
        print(f"Loaded audio: {len(audio_data)/sr:.2f}s at {sr}Hz")
        
        # ==================== TIER 1: INITIAL EVENT DETECTION ====================
        print("\n--- TIER 1: Initial Event Detection ---")
        
        # Rule 1: Energy Threshold Detection
        tier1_result = _tier1_energy_threshold(audio_data, sr, config)
        if not tier1_result['passed']:
            return {
                'is_explosion': False,
                'reason': tier1_result['reason'],
                'file_path': audio_path,
                'sample_rate': sr,
                'tier1_result': tier1_result
            }
        
        candidates = tier1_result['candidates']
        print(f"Found {len(candidates)} high-energy candidates")
        
        # Rule 2: Broadband Check
        broadband_candidates = []
        for candidate in candidates:
            if _tier1_broadband_check(audio_data, sr, candidate, config):
                broadband_candidates.append(candidate)
        
        if not broadband_candidates:
            return {
                'is_explosion': False,
                'reason': 'No broadband high-energy transients found',
                'file_path': audio_path,
                'sample_rate': sr,
                'tier1_result': tier1_result
            }
            
        print(f"Found {len(broadband_candidates)} broadband candidates")
        
        # ==================== TIER 2: PRIMARY CLASSIFICATION ====================
        print("\n--- TIER 2: Primary Classification ---")
        
        for i, candidate in enumerate(broadband_candidates):
            print(f"\nAnalyzing candidate {i+1}/{len(broadband_candidates)}")
            
            # Extract candidate segment
            start_idx = candidate['start_idx']
            end_idx = candidate['end_idx']
            segment = audio_data[start_idx:end_idx]
            
            # Rule 3: Shockwave Detection
            shockwave_result = _tier2_shockwave_detection(segment, sr, config)
            if not shockwave_result['valid']:
                print(f"  ❌ Candidate {i+1}: {shockwave_result['reason']}")
                continue
                
            print(f"  ✓ Valid shockwave detected (rise time: {shockwave_result['rise_time_ms']:.1f}ms)")
            
            # Rule 4: Bubble Pulse Search
            bubble_result = _tier2_bubble_pulse_search(segment, sr, shockwave_result, config)
            if not bubble_result['found']:
                print(f"  ❌ Candidate {i+1}: {bubble_result['reason']}")
                continue
                
            print(f"  ✓ Bubble pulse found (period: {bubble_result['period_ms']:.1f}ms)")
            
            # Rule 5: Physical Plausibility Check
            if not _tier2_physical_plausibility(bubble_result, config):
                print(f"  ❌ Candidate {i+1}: Bubble period outside physically plausible range")
                continue
            
            print(f"  ✓ Physical plausibility confirmed")
            
            # If we reach here, we have a valid explosion detection!
            explosion_time = (start_idx + shockwave_result['peak_idx']) / sr
            
            return {
                'is_explosion': True,
                'reason': 'Shockwave and plausible bubble pulse detected',
                'file_path': audio_path,
                'sample_rate': sr,
                'explosion_time_s': explosion_time,
                'metrics': {
                    'shockwave_time_s': explosion_time,
                    'bubble_period_ms': bubble_result['period_ms'],
                    'rise_time_ms': shockwave_result['rise_time_ms'],
                    'shockwave_amplitude': shockwave_result['peak_amplitude'],
                    'bubble_amplitude': bubble_result['bubble_amplitude'],
                    'amplitude_ratio': bubble_result['amplitude_ratio']
                },
                'detection_data': {
                    'audio_segment': segment,
                    'segment_start_time': start_idx / sr,
                    'shockwave_idx': shockwave_result['peak_idx'],
                    'bubble_idx': bubble_result['bubble_idx'],
                    'sample_rate': sr
                }
            }
        
        # No valid explosions found
        return {
            'is_explosion': False,
            'reason': 'No candidates passed all explosion verification tests',
            'file_path': audio_path,
            'sample_rate': sr,
            'candidates_analyzed': len(broadband_candidates)
        }
        
    except Exception as e:
        return {
            'is_explosion': False,
            'reason': f'Error analyzing audio: {str(e)}',
            'file_path': audio_path
        }


def _tier1_energy_threshold(audio_data, sr, config):
    """
    Tier 1, Rule 1: Energy Threshold Detection
    Finds segments where acoustic energy dramatically exceeds baseline.
    """
    
    # Calculate short-term RMS energy
    frame_length = int(config["frame_length_ms"] * sr / 1000)
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=frame_length//2)[0]
    
    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-8)  # Add small value to avoid log(0)
    
    # Calculate baseline (ambient noise level)
    baseline_db = np.median(rms_db)
    
    # Find high-energy segments
    threshold_db = baseline_db + config["energy_threshold_db"]
    high_energy_frames = np.where(rms_db > threshold_db)[0]
    
    if len(high_energy_frames) == 0:
        return {
            'passed': False,
            'reason': 'No high-energy transient found',
            'baseline_db': baseline_db,
            'threshold_db': threshold_db
        }
    
    # Group consecutive frames into candidates
    candidates = []
    hop_length = frame_length // 2
    
    # Find continuous regions
    frame_groups = []
    current_group = [high_energy_frames[0]]
    
    for frame in high_energy_frames[1:]:
        if frame == current_group[-1] + 1:
            current_group.append(frame)
        else:
            frame_groups.append(current_group)
            current_group = [frame]
    frame_groups.append(current_group)
    
    # Convert frame groups to sample indices
    for group in frame_groups:
        start_frame = group[0]
        end_frame = group[-1]
        
        start_idx = max(0, start_frame * hop_length - frame_length)
        end_idx = min(len(audio_data), (end_frame + 1) * hop_length + frame_length)
        
        # Check minimum duration
        duration_ms = (end_idx - start_idx) / sr * 1000
        if duration_ms >= config["min_event_duration_ms"]:
            candidates.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration_ms': duration_ms,
                'max_energy_db': np.max(rms_db[start_frame:end_frame+1])
            })
    
    return {
        'passed': len(candidates) > 0,
        'candidates': candidates,
        'baseline_db': baseline_db,
        'threshold_db': threshold_db,
        'reason': f'Found {len(candidates)} energy candidates' if candidates else 'No candidates meet minimum duration'
    }


def _tier1_broadband_check(audio_data, sr, candidate, config):
    """
    Tier 1, Rule 2: Broadband Check
    Verifies that significant energy is present across multiple frequency bands.
    """
    
    # Extract candidate segment
    segment = audio_data[candidate['start_idx']:candidate['end_idx']]
    
    # Compute FFT
    fft = np.fft.fft(segment)
    freqs = np.fft.fftfreq(len(segment), 1/sr)
    magnitude = np.abs(fft)
    
    # Only consider positive frequencies
    positive_freq_mask = freqs >= 0
    freqs = freqs[positive_freq_mask]
    magnitude = magnitude[positive_freq_mask]
    
    total_energy = np.sum(magnitude**2)
    
    # Check energy in each frequency band
    bands_passed = 0
    for low_freq, high_freq in config["broadband_check_bands"]:
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_energy = np.sum(magnitude[band_mask]**2)
        band_energy_ratio = band_energy / total_energy
        
        if band_energy_ratio >= config["broadband_energy_threshold"]:
            bands_passed += 1
    
    # Require energy in at least 2 out of 3 frequency bands
    return bands_passed >= 2


def _tier2_shockwave_detection(segment, sr, config):
    """
    Tier 2, Rule 3: Shockwave Detection
    Analyzes the segment for extremely short rise time characteristic of shockwaves.
    """
    
    # Find the absolute peak (potential shockwave)
    peak_idx = np.argmax(np.abs(segment))
    peak_amplitude = np.abs(segment[peak_idx])
    
    # Analyze rise time leading up to the peak
    # Look backwards from peak to find 10% and 90% amplitude points
    search_start = max(0, peak_idx - int(0.1 * sr))  # Search up to 100ms before peak
    search_segment = segment[search_start:peak_idx+1]
    search_amplitudes = np.abs(search_segment)
    
    # Find 10% and 90% amplitude points
    amp_10_percent = 0.1 * peak_amplitude
    amp_90_percent = 0.9 * peak_amplitude
    
    # Find last point below 10% threshold
    below_10_mask = search_amplitudes < amp_10_percent
    if not np.any(below_10_mask):
        return {
            'valid': False,
            'reason': 'Cannot find 10% amplitude point for rise time calculation'
        }
    
    idx_10_percent = np.where(below_10_mask)[0][-1]  # Last point below 10%
    
    # Find first point above 90% threshold after the 10% point
    above_90_mask = search_amplitudes[idx_10_percent:] > amp_90_percent
    if not np.any(above_90_mask):
        return {
            'valid': False,
            'reason': 'Cannot find 90% amplitude point for rise time calculation'
        }
    
    idx_90_percent = idx_10_percent + np.where(above_90_mask)[0][0]
    
    # Calculate rise time
    rise_time_samples = idx_90_percent - idx_10_percent
    rise_time_ms = (rise_time_samples / sr) * 1000
    
    # Check if rise time is sufficiently short for a shockwave
    if rise_time_ms > config["shockwave_rise_time_ms"]:
        return {
            'valid': False,
            'reason': f'Rise time too slow ({rise_time_ms:.1f}ms > {config["shockwave_rise_time_ms"]}ms threshold)'
        }
    
    return {
        'valid': True,
        'peak_idx': peak_idx,
        'peak_amplitude': peak_amplitude,
        'rise_time_ms': rise_time_ms,
        'reason': f'Valid shockwave detected (rise time: {rise_time_ms:.1f}ms)'
    }


def _tier2_bubble_pulse_search(segment, sr, shockwave_result, config):
    """
    Tier 2, Rule 4: Bubble Pulse Search
    Searches for secondary peak (bubble pulse) after the shockwave.
    """
    
    shockwave_idx = shockwave_result['peak_idx']
    shockwave_amplitude = shockwave_result['peak_amplitude']
    
    # Define search window after shockwave
    search_start_idx = shockwave_idx + int(config["bubble_pulse_min_period_ms"] * sr / 1000)
    search_end_idx = min(len(segment), 
                        shockwave_idx + int(config["bubble_pulse_search_window_ms"] * sr / 1000))
    
    if search_start_idx >= search_end_idx or search_start_idx >= len(segment):
        return {
            'found': False,
            'reason': 'Search window too short or extends beyond segment'
        }
    
    search_segment = segment[search_start_idx:search_end_idx]
    
    # Find peaks in the search region
    min_height = config["bubble_pulse_amplitude_ratio"] * shockwave_amplitude
    prominence = config["peak_prominence_factor"] * min_height
    
    peaks, properties = scipy.signal.find_peaks(
        np.abs(search_segment), 
        height=min_height,
        prominence=prominence
    )
    
    if len(peaks) == 0:
        return {
            'found': False,
            'reason': f'No secondary peaks found above {config["bubble_pulse_amplitude_ratio"]*100:.0f}% of shockwave amplitude'
        }
    
    # Select the most prominent peak as bubble pulse
    peak_heights = properties['peak_heights']
    most_prominent_idx = np.argmax(peak_heights)
    bubble_peak_idx = peaks[most_prominent_idx]
    bubble_amplitude = peak_heights[most_prominent_idx]
    
    # Convert to absolute index in segment
    bubble_idx_absolute = search_start_idx + bubble_peak_idx
    
    # Calculate bubble period
    period_samples = bubble_idx_absolute - shockwave_idx
    period_ms = (period_samples / sr) * 1000
    
    return {
        'found': True,
        'bubble_idx': bubble_idx_absolute,
        'bubble_amplitude': bubble_amplitude,
        'period_ms': period_ms,
        'amplitude_ratio': bubble_amplitude / shockwave_amplitude,
        'reason': f'Bubble pulse found at {period_ms:.1f}ms after shockwave'
    }


def _tier2_post_peak_decay_check(audio_data, sr, candidate_start_idx, shockwave_result, config):
    """
    Tier 2, Rule 5b: Post-Peak Decay Check
    Real underwater explosions decay rapidly after the shockwave peak.
    False positives (boat noise, etc.) maintain high energy after the peak.
    """
    peak_idx_abs = candidate_start_idx + shockwave_result['peak_idx']
    window_samples = int(config["post_peak_window_ms"] * sr / 1000)
    
    post_start = peak_idx_abs
    post_end = min(len(audio_data), peak_idx_abs + window_samples)
    
    if post_end <= post_start:
        return {'passed': False, 'reason': 'Post-peak window extends beyond audio', 'post_peak_rms': None}
    
    post_segment = audio_data[post_start:post_end]
    post_rms = float(np.sqrt(np.mean(post_segment**2)))
    
    passed = post_rms <= config["post_peak_max_rms"]
    return {
        'passed': passed,
        'post_peak_rms': round(post_rms, 4),
        'reason': f'Post-peak RMS={post_rms:.4f} {"<=" if passed else ">"} {config["post_peak_max_rms"]}'
    }


def _tier2_physical_plausibility(bubble_result, config):
    """
    Tier 2, Rule 5: Physical Plausibility Check
    Verifies that bubble period falls within physically realistic range.
    """
    
    period_ms = bubble_result['period_ms']
    min_period = config["bubble_pulse_min_period_ms"]
    max_period = config["bubble_pulse_max_period_ms"]
    
    return min_period <= period_ms <= max_period


def plot_detection_details(audio_data, sr, metrics):
    """
    Creates a two-panel visualization showing the detected explosion signature.
    
    Args:
        audio_data (np.array): Audio data
        sr (int): Sample rate  
        metrics (dict): Detection metrics and data
    """
    
    detection_data = metrics['detection_data']
    segment = detection_data['audio_segment']
    shockwave_idx = detection_data['shockwave_idx'] 
    bubble_idx = detection_data['bubble_idx']
    segment_start_time = detection_data['segment_start_time']
    
    # Create time arrays
    segment_time = np.arange(len(segment)) / sr + segment_start_time
    full_time = np.arange(len(audio_data)) / sr
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Dynamite Explosion Detection Analysis', fontsize=16, fontweight='bold')
    
    # ==================== WAVEFORM PLOT ====================
    ax1.plot(segment_time, segment, 'b-', linewidth=1, alpha=0.7, label='Audio Waveform')
    
    # Mark shockwave peak
    shockwave_time = segment_time[shockwave_idx]
    ax1.axvline(shockwave_time, color='red', linewidth=2, linestyle='--', alpha=0.8)
    ax1.plot(shockwave_time, segment[shockwave_idx], 'ro', markersize=8, label='Shockwave Peak')
    ax1.text(shockwave_time, segment[shockwave_idx] + 0.1*np.max(np.abs(segment)), 
             'Shockwave Peak', ha='center', va='bottom', fontweight='bold', color='red')
    
    # Mark bubble pulse peak
    bubble_time = segment_time[bubble_idx]
    ax1.axvline(bubble_time, color='orange', linewidth=2, linestyle='--', alpha=0.8)
    ax1.plot(bubble_time, segment[bubble_idx], 'o', color='orange', markersize=8, label='Bubble Pulse')
    ax1.text(bubble_time, segment[bubble_idx] + 0.1*np.max(np.abs(segment)), 
             'First Bubble Pulse', ha='center', va='bottom', fontweight='bold', color='orange')
    
    # Add double-headed arrow for bubble period
    arrow_y = 0.8 * np.max(np.abs(segment))
    ax1.annotate('', xy=(bubble_time, arrow_y), xytext=(shockwave_time, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text((shockwave_time + bubble_time) / 2, arrow_y + 0.05*np.max(np.abs(segment)),
             f'Bubble Period (tpuls)\n{metrics["metrics"]["bubble_period_ms"]:.1f}ms', 
             ha='center', va='bottom', fontweight='bold', color='purple',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Amplitude', fontweight='bold')
    ax1.set_title('Annotated Waveform - Explosion Event Detection', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ==================== SPECTROGRAM PLOT ====================
    # Compute spectrogram
    D = librosa.stft(segment)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', 
                                   ax=ax2, cmap='viridis')
    
    # Adjust time axis to match segment timing
    ax2_xlim = ax2.get_xlim()
    time_offset = segment_start_time
    ax2.set_xlim([ax2_xlim[0] + time_offset, ax2_xlim[1] + time_offset])
    
    # Mark explosion time
    ax2.axvline(shockwave_time, color='red', linewidth=3, linestyle='-', alpha=0.9)
    ax2.text(shockwave_time + 0.01, ax2.get_ylim()[1] * 0.9, 
             'Broadband Shockwave Event', rotation=90, ha='left', va='top', 
             fontweight='bold', color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add bracket for reverberant tail
    tail_start = bubble_time
    tail_end = segment_time[-1]
    tail_y = ax2.get_ylim()[1] * 0.7
    
    # Draw bracket
    bracket_height = ax2.get_ylim()[1] * 0.05
    ax2.plot([tail_start, tail_start], [tail_y - bracket_height, tail_y + bracket_height], 
             'k-', linewidth=2)
    ax2.plot([tail_end, tail_end], [tail_y - bracket_height, tail_y + bracket_height], 
             'k-', linewidth=2)  
    ax2.plot([tail_start, tail_end], [tail_y, tail_y], 'k-', linewidth=2)
    
    ax2.text((tail_start + tail_end) / 2, tail_y + bracket_height * 2,
             'Reverberant Tail', ha='center', va='bottom', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_title('Annotated Spectrogram - Frequency Domain Analysis', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.set_label('Magnitude (dB)', fontweight='bold')
    
    # Add detection summary text box
    summary_text = f"""Detection Summary:
    • Explosion detected at {metrics['explosion_time_s']:.2f}s
    • Shockwave rise time: {metrics['metrics']['rise_time_ms']:.1f}ms  
    • Bubble period: {metrics['metrics']['bubble_period_ms']:.1f}ms
    • Amplitude ratio: {metrics['metrics']['amplitude_ratio']:.2f}
    • Classification: POSITIVE DETECTION"""
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function for command-line usage of the dynamite detector.
    """
    
    parser = argparse.ArgumentParser(
        description='Dynamite Fishing Detection System - Acoustic Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dynamite_detector.py audio_file.wav
  python dynamite_detector.py --config custom_config.json audio_file.wav
  python dynamite_detector.py --plot --save-plot explosion_analysis.png audio_file.wav

Based on: "Acoustic Signatures of Underwater Explosions: A Technical Report"
        """
    )
    
    parser.add_argument('audio_file', help='Path to the audio file (.wav)')
    parser.add_argument('--config', help='Path to custom configuration JSON file')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate visualization plot for positive detections')
    parser.add_argument('--save-plot', metavar='FILENAME',
                       help='Save plot to specified filename (implies --plot)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    # Load custom config if provided
    config = DETECTOR_CONFIG.copy()
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            config.update(custom_config)
            print(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    print("=" * 60)
    print("DYNAMITE FISHING DETECTION SYSTEM")
    print("=" * 60)
    print(f"Analyzing: {args.audio_file}")
    print(f"Configuration: {len(config)} parameters loaded")
    print("=" * 60)
    
    # Analyze the audio file
    results = analyze_audio_for_explosion(args.audio_file, config)
    
    # Print results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"File: {results['file_path']}")
    print(f"Detection Status: {'EXPLOSION DETECTED' if results['is_explosion'] else 'NO EXPLOSION DETECTED'}")
    print(f"Reason: {results['reason']}")
    
    if results['is_explosion']:
        print(f"\n📍 EXPLOSION DETAILS:")
        print(f"   • Time: {results['explosion_time_s']:.3f} seconds")
        print(f"   • Shockwave rise time: {results['metrics']['rise_time_ms']:.1f}ms")
        print(f"   • Bubble period: {results['metrics']['bubble_period_ms']:.1f}ms") 
        print(f"   • Shockwave amplitude: {results['metrics']['shockwave_amplitude']:.4f}")
        print(f"   • Bubble amplitude: {results['metrics']['bubble_amplitude']:.4f}")
        print(f"   • Amplitude ratio: {results['metrics']['amplitude_ratio']:.3f}")
        
        # Generate plot if requested
        if args.plot or args.save_plot:
            print(f"\n🎨 GENERATING VISUALIZATION...")
            try:
                # Load audio data for plotting
                audio_data, sr = librosa.load(args.audio_file, sr=config["sample_rate"])
                fig = plot_detection_details(audio_data, sr, results)
                
                if args.save_plot:
                    fig.savefig(args.save_plot, dpi=300, bbox_inches='tight')
                    print(f"   Plot saved to: {args.save_plot}")
                else:
                    plt.show()
                    
            except Exception as e:
                print(f"   Error generating plot: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Return appropriate exit code
    sys.exit(0 if results['is_explosion'] else 1)


if __name__ == "__main__":
    main()
