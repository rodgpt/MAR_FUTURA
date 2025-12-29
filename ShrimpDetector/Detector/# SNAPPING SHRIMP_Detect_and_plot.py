# Detect_and_plot_with_filter.py
# Author: Margherita Silvestri
# Date: 03-10-25
# Description: Snapping shrimp click detector using envelope + derivative thresholds
#              with 2 kHz high-pass filter, inter-click interval dead time,
#              and corrupted file handling

# =============================================================================
# REQUIRED PACKAGES
# =============================================================================
# Create conda environment and install: numpy, scipy, matplotlib
# Command: conda create -n snaprate python=3.9 numpy scipy matplotlib
#          conda activate snaprate

import os                                    # File path operations
import glob                                  # Pattern matching for file search
import numpy as np                           # Array operations and numerical computing
import scipy.io.wavfile as wav              # WAV file I/O
from scipy.signal import hilbert, butter, filtfilt  # Signal processing tools
import matplotlib.pyplot as plt              # Visualization


###############################################################################
# FUNCTION 1: Safe WAV Reading with Error Handling
###############################################################################
def safe_read_wav(path):
    """
    Attempt to read a WAV file; skip if corrupted or unsupported format.
    
    Parameters:
    -----------
    path : str
        Full file path to the WAV file
        
    Returns:
    --------
    tuple : (sample_rate, audio_data) on success, or (None, None) on failure
    """
    try:
        # Read WAV file using scipy
        return wav.read(path)
    except ValueError as e:
        # Handle corrupted or non-standard WAV formats
        print(f"⚠️  Skipping unsupported/corrupted file: {os.path.basename(path)} ({e})")
        return None, None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"❌ Error reading {os.path.basename(path)}: {e}")
        return None, None


###############################################################################
# FUNCTION 2: High-Pass Filter (Butterworth, Zero-Phase)
###############################################################################
def highpass_filter(sig, sr, cutoff=2000, order=4):
    """
    Apply zero-phase Butterworth high-pass filter to isolate frequencies > cutoff.
    
    Removes low-frequency noise (waves, vessels, ambient rumble) while preserving
    high-frequency shrimp click energy (typically 2-20 kHz).
    
    Parameters:
    -----------
    sig : numpy.ndarray
        1D audio signal (normalized, zero-mean)
    sr : int
        Sample rate in Hz (e.g., 96000)
    cutoff : int, optional
        Cutoff frequency in Hz (default: 2000)
    order : int, optional
        Filter order; higher = steeper rolloff (default: 4)
        
    Returns:
    --------
    numpy.ndarray : Filtered signal (same length as input)
    """
    nyq = 0.5 * sr                          # Nyquist frequency (max representable frequency)
    norm_cutoff = cutoff / nyq              # Normalize cutoff to [0, 1] for scipy
    b, a = butter(order, norm_cutoff, btype='high')  # Design filter coefficients
    return filtfilt(b, a, sig)              # Zero-phase filtering (forward + backward pass)


###############################################################################
# FUNCTION 3: Main Click Detection and Plotting Function
###############################################################################
def detect_and_plot_with_ici(wav_path, output_dir=None, env_thresh_factor=2.5, 
                              der_thresh_factor=2.5, t_min=0.03):
    """
    Detect snapping shrimp clicks using dual-threshold method (envelope + derivative).
    
    Processing Pipeline:
    --------------------
    1. Read and preprocess audio (mono conversion, normalization, DC removal)
    2. High-pass filter (>2 kHz) to remove low-frequency noise
    3. Compute Hilbert envelope and derivative to track amplitude changes
    4. Set adaptive thresholds based on signal statistics
    5. Detect clicks using dual thresholds + refractory period
    6. Save results (labels, scores) and generate 5-panel diagnostic figure
    
    Parameters:
    -----------
    wav_path : str
        Full path to input WAV file
    output_dir : str, optional
        Output directory (default: current working directory)
    env_thresh_factor : float, optional
        Envelope threshold multiplier (default: 2.5)
        Formula: threshold = mean + factor × std
    der_thresh_factor : float, optional
        Derivative threshold multiplier (default: 2.5)
    t_min : float, optional
        Minimum inter-click interval in seconds (default: 0.04 = 40 ms)
        Acts as refractory period to prevent double-counting
    
    Outputs:
    --------
    - {basename}.labels.txt : Audacity-compatible label file (start, end, label, score)
    - {basename}.scores.txt : Detection scores for ROC analysis (time, score, label)
    - {basename}_overview.png : 5-panel diagnostic figure (envelope, derivative, rates, ICI)
    """
    
    # =========================================================================
    # STEP 1: Prepare Output File Paths
    # =========================================================================
    out = output_dir or os.getcwd()          # Use specified dir or current directory
    os.makedirs(out, exist_ok=True)          # Create output dir if it doesn't exist
    base = os.path.splitext(os.path.basename(wav_path))[0]  # Extract filename without extension
    
    # Define three output file paths
    labels_file = os.path.join(out, f"{base}.labels.txt")   # Audacity labels
    scores_file = os.path.join(out, f"{base}.scores.txt")   # Detection scores
    fig_file = os.path.join(out, f"{base}_overview.png")    # Diagnostic figure

    # =========================================================================
    # STEP 2: Read WAV File
    # =========================================================================
    sr, data = safe_read_wav(wav_path)       # sr = sample rate, data = audio samples
    if data is None:                          # Skip if file reading failed
        return

    # =========================================================================
    # STEP 3: Convert Stereo to Mono
    # =========================================================================
    if data.ndim > 1:                        # Check if stereo (2D array)
        data = data[:, 0]                     # Take only first channel (left)

    # =========================================================================
    # STEP 4: Normalize Amplitude to [-1, 1] Range
    # =========================================================================
    # Purpose: Standardize signal amplitude across recordings with different:
    #   - Recording gains
    #   - Hydrophone sensitivities
    #   - Source distances
    #   - Bit depths (16-bit, 24-bit, 32-bit)
    
    data = data.astype(np.float32)           # Convert integer samples to float32
    max_amplitude = np.max(np.abs(data))     # Find peak absolute value (handles +/- peaks)
    data /= (max_amplitude + 1e-10)          # Scale to [-1, 1]; epsilon prevents div by zero

    # =========================================================================
    # STEP 5: Remove DC Offset (Center Signal at Zero)
    # =========================================================================
    # Purpose: Eliminate constant bias introduced by:
    #   - ADC/preamp electronics
    #   - Sensor drift
    #   - Environmental electrical interference
    # Effect: Improves filter performance and envelope calculation accuracy
    
    dc_offset = np.mean(data)                # Calculate average amplitude (DC component)
    sig = data - dc_offset                   # Subtract mean to center waveform at zero

    # =========================================================================
    # STEP 6: Apply High-Pass Filter (>2 kHz)
    # =========================================================================
    # Purpose: Remove low-frequency noise sources:
    #   - Ocean waves (< 1 Hz)
    #   - Vessel engines (50-500 Hz)
    #   - Wind/flow noise (< 1 kHz)
    # Rationale: Shrimp clicks have peak energy at 2-20 kHz
    
    sig = highpass_filter(sig, sr, cutoff=2000, order=4)  # Filter out frequencies < 2 kHz

    # =========================================================================
    # STEP 7: Compute Envelope and Derivative
    # =========================================================================
    # Hilbert Transform Method:
    # - Creates analytic signal: z(t) = x(t) + j·H{x(t)}
    # - Envelope = |z(t)| tracks instantaneous amplitude
    # - Derivative = d|z(t)|/dt highlights rapid amplitude rises
    # Advantage: Separates transient clicks from steady-state noise
    
    env = np.abs(hilbert(sig))               # Compute Hilbert envelope (amplitude modulation)
    denv = np.diff(env, prepend=env[0])      # Compute derivative; prepend maintains array length

    # =========================================================================
    # STEP 8: Compute Adaptive Thresholds
    # =========================================================================
    # Adaptive Method: threshold = mean + factor × std
    # Why adaptive? Noise levels vary across:
    #   - Recording sites
    #   - Weather conditions
    #   - Times of day
    # Result: Thresholds automatically scale to local noise statistics
    
    e_th = env.mean() + env.std() * env_thresh_factor    # Envelope threshold
    d_th = denv.mean() + denv.std() * der_thresh_factor  # Derivative threshold

    # =========================================================================
    # STEP 9: Convert Minimum Inter-Click Interval to Samples
    # =========================================================================
    # Refractory Period: Prevents detecting the same click multiple times
    # Example: At 96 kHz, 0.04 s = 3840 samples
    
    min_samp = int(t_min * sr)               # Convert time (seconds) to samples

    # =========================================================================
    # STEP 10: Detect Clicks (Dual Threshold + Refractory Period)
    # =========================================================================
    # Detection Logic:
    #   1. Envelope exceeds threshold (loud enough)
    #   2. Derivative exceeds threshold (sharp/impulsive enough)
    #   3. Sufficient time since last detection (prevents double-counting)
    
    clicks = []                              # List to store click times (seconds)
    clicks_scores = []                       # List to store click amplitudes
    last_click_idx = -min_samp               # Initialize to allow first detection
    
    for i in range(len(denv)):               # Loop through all samples
        if (env[i] > e_th and                # Condition 1: Loud
            denv[i] > d_th and               # Condition 2: Sharp
            (i - last_click_idx >= min_samp)):  # Condition 3: Refractory period elapsed
            clicks.append(i / sr)             # Convert sample index to time (seconds)
            clicks_scores.append(env[i])      # Store envelope value as detection score
            last_click_idx = i                # Update last detection index
    
    clicks = np.array(clicks)                # Convert list to numpy array
    clicks_scores = np.array(clicks_scores)  # Convert list to numpy array

    # =========================================================================
    # STEP 11: Save Detection Results to Text Files
    # =========================================================================
    # Output Format:
    #   - labels.txt: Audacity-compatible (start, end, label, score)
    #   - scores.txt: ROC analysis format (time, score, label)
    
    with open(labels_file, 'w') as f, open(scores_file, 'w') as sf:
        # Write header rows
        f.write("# start\tend\tlabel\tscore\n")
        sf.write("# time\tscore\tlabel\n")
        
        # Loop through detections
        for idx, (t, score) in enumerate(zip(clicks, clicks_scores), 1):
            # Labels file: start, end (1 ms duration), label, score
            f.write(f"{t:.6f}\t{t+0.001:.6f}\tClick_{idx}\t{score:.6f}\n")
            # Scores file: time, score, label
            sf.write(f"{t:.6f}\t{score:.6f}\tClick_{idx}\n")

    # =========================================================================
    # STEP 12: Generate Diagnostic Plots
    # =========================================================================
    
    # --- Compute Metrics ---
    total_time = len(sig) / sr               # Recording duration (seconds)
    secs = int(np.ceil(total_time))          # Round up to whole seconds
    time_env = np.linspace(0, total_time, len(env))  # Time axis for envelope
    
    # Compute clicks per second (1-second bins)
    cps = [np.sum((clicks >= s) & (clicks < s+1)) for s in range(secs)]
    cumulative = np.cumsum(cps)              # Cumulative click count over time
    
    # Compute inter-click intervals (ICI)
    icis = np.diff(clicks)                   # Time difference between consecutive clicks
    times_ici = clicks[1:]                   # Time points for ICI values (one less than clicks)
    
    # Calculate mode ICI (most common inter-click interval)
    if len(icis) > 0:
        counts, bins = np.histogram(icis, bins=100)  # Create histogram
        mode_idx = counts.argmax()           # Find bin with max count
        mode_ici = (bins[mode_idx] + bins[mode_idx+1]) / 2  # Calculate bin center
    else:
        mode_ici = 0                         # No ICI if 0 or 1 clicks detected

    # --- Create Figure ---
    plt.figure(figsize=(14, 12))             # Create figure (width, height in inches)

    # --- Panel 1: Envelope + Detected Clicks (First 2 Seconds) ---
    ax1 = plt.subplot(5, 1, 1)               # First subplot in 5×1 grid
    lim = min(len(env), sr * 2)              # Display limit: 2 seconds or file end
    t_disp = time_env[:lim]                  # Time axis for display window
    ax1.plot(t_disp, env[:lim], 'b-', label='Envelope')  # Plot envelope in blue
    mask = clicks <= t_disp[-1]              # Boolean mask for clicks in display window
    ax1.scatter(clicks[mask], env[(clicks[mask]*sr).astype(int)],  # Plot clicks as red dots
                c='r', s=30, label='Clicks')
    ax1.set_ylabel('Envelope')               # Y-axis label
    ax1.legend(loc='upper right')            # Add legend
    ax1.set_title(base)                      # Title with filename

    # --- Panel 2: Derivative + Threshold Line ---
    ax2 = plt.subplot(5, 1, 2)               # Second subplot
    ax2.plot(t_disp, denv[:lim], 'g-', label='Derivative')  # Plot derivative in green
    ax2.axhline(d_th, color='r', linestyle='--', label='Threshold')  # Horizontal threshold line
    ax2.set_ylabel('dEnv/dt')                # Y-axis label (derivative notation)
    ax2.legend(loc='upper right')            # Add legend

    # --- Panel 3: Clicks Per Second ---
    ax3 = plt.subplot(5, 1, 3)               # Third subplot
    xs = np.arange(1, secs+1)                # X-axis: seconds (1 to total)
    ax3.plot(xs, cps, 'b-o')                 # Line plot with circle markers
    ax3.fill_between(xs, cps, alpha=0.3)     # Shaded area under curve
    ax3.axhline(np.mean(cps), color='r', linestyle='--',  # Mean line
                label=f"Mean: {np.mean(cps):.1f} clicks/sec")
    ax3.set_ylabel('Clicks/sec')             # Y-axis label
    ax3.set_xlim(1, secs)                    # Set X-axis limits
    ax3.legend()                             # Add legend

    # --- Panel 4: Cumulative Clicks ---
    ax4 = plt.subplot(5, 1, 4)               # Fourth subplot
    ax4.plot(xs, cumulative, 'r-o')          # Line plot in red
    ax4.fill_between(xs, cumulative, alpha=0.3, color='red')  # Shaded area
    ax4.set_ylabel('Cumulative clicks')      # Y-axis label
    ax4.set_xlim(1, secs)                    # Set X-axis limits

    # --- Panel 5: Inter-Click Interval (ICI) Scatter ---
    ax5 = plt.subplot(5, 1, 5)               # Fifth subplot
    ax5.scatter(times_ici, icis, c='m', s=20)  # Scatter plot in magenta
    ax5.axhline(mode_ici, color='k', linestyle='--',  # Mode ICI line
                label=f"Mode ICI: {mode_ici:.4f}s")
    ax5.set_xlabel('Time (s)')               # X-axis label
    ax5.set_ylabel('Inter-click interval (s)')  # Y-axis label
    ax5.set_title('Raw ICI vs Time')         # Subplot title
    ax5.legend(loc='upper right')            # Add legend
    ax5.grid(alpha=0.3)                      # Add grid with transparency

    # --- Save and Close ---
    plt.tight_layout()                       # Auto-adjust spacing to prevent overlap
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')  # Save at high resolution
    plt.close()                              # Close figure to free memory

    # --- Print Summary ---
    print(f"Processed {base}:")
    print(f"  Labels → {labels_file}")
    print(f"  Scores → {scores_file}")
    print(f"  Figure → {fig_file}")


###############################################################################
# MAIN EXECUTION BLOCK (Batch Processing)
###############################################################################
if __name__ == "__main__":
    """
    Batch processing workflow:
    1. Scan input folder for all WAV files
    2. Create output subdirectory for each file
    3. Run detection on each file with specified parameters
    
    This block only executes when script is run directly (not when imported).
    """
    
    # --- Define Input/Output Paths ---
    # MODIFY THIS PATH to match your data location
    input_folder = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS\LASCRUCES"
    
    # Create main output directory
    output_folder = os.path.join(input_folder, "output_all_8")
    os.makedirs(output_folder, exist_ok=True)  # Create if doesn't exist

    # --- Find All WAV Files ---
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))  # Pattern match *.wav
    
    # --- Process Each File ---
    for wav_path in wav_files:
        # Extract filename without extension
        base = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Create subdirectory for this file's outputs
        out_dir = os.path.join(output_folder, base)
        os.makedirs(out_dir, exist_ok=True)  # Create subdirectory
        
        # Run detector with parameters
        detect_and_plot_with_ici(
            wav_path=wav_path,              # Input WAV file
            output_dir=out_dir,             # Output directory for this file
            env_thresh_factor=2.5,          # Envelope threshold: mean + 2.5×std
            der_thresh_factor=2.5,          # Derivative threshold: mean + 2.5×std
            t_min=0.03                    # Minimum inter-click interval: 30 ms
        )

# =============================================================================
# END OF SCRIPT
# =============================================================================
