import os
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample
from copy import deepcopy
from tqdm import tqdm
import pyloudnorm as pyln

# Global settings - Modify Here
OUTPUT_DIR = 'textrolmix'
MODE = 'min'  # Options: 'min' or 'max'
SAMPLING_FREQUENCY = 8000  # Options: 8000 or 16000


# Helper Functions
def root_mean_square(signal):
    """Calculate the root mean square of a signal."""
    return np.sqrt(np.mean(signal**2))

def signal_to_noise_ratio(signal, noise):
    """Calculate the signal-to-noise ratio (SNR) in dB."""
    return 20 * np.log10(root_mean_square(signal) / root_mean_square(noise))

def normalize_rms(signal, target_rms):
    """Normalize signal to the target RMS."""
    scale_factor = target_rms / root_mean_square(signal)
    return signal * scale_factor, scale_factor

def normalize_loudness(signal, samplerate, target_loudness):
    """Normalize the loudness of an audio signal to the target loudness."""
    meter = pyln.Meter(samplerate)
    loudness = meter.integrated_loudness(signal)
    return pyln.normalize.loudness(signal, loudness, target_loudness)

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """Detects leading silence by checking if the RMS of audio chunks falls below a threshold. """
    # Calculate the RMS of chunks
    for i in range(0, len(sound), chunk_size):
        # Calculate RMS of the chunk
        rms = 20 * np.log10(np.sqrt(np.mean(sound[i:i+chunk_size]**2)))
        if rms > silence_threshold:
            # The index of the first sound sample above the silence threshold.
            return max(0, i-chunk_size)
    return 0  # Return 0 if the whole file is silent


def preprocess_single_audio(audio_fp, target_sr, target_loudness=None):
    '''Read individual utterance, resample and normalize loudness if necessary'''
    file_path = os.path.join('textrolspeech', audio_fp)
    data, samplerate = sf.read(file_path)
    # Trim beginning silence, otherwise would affect mixture SNR
    start_index = detect_leading_silence(data, -40, 10)
    data = data[start_index:]
         
    if data.ndim > 1:
        data = data.mean(axis=1)
    if samplerate != target_sr:
        data = resample(data, int(len(data) * target_sr / samplerate))
    
    # save to "single" folder
    file_path_to_save = os.path.join(f'{OUTPUT_DIR}/single_{target_sr}Hz', audio_fp)
    directory_path = os.path.dirname(file_path_to_save) 
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if not os.path.exists(file_path_to_save):
        sf.write(file_path_to_save, data, target_sr)
    
    # normalize loudness to generate mixture
    if target_loudness:
        data = normalize_loudness(data, target_sr, target_loudness)

    return data, target_sr



def mix_speeches(target_speech, background_speeches, snr_db, mode):
    # Find the minimum or maximum length to match the mode of operation
    if mode == 'min':
        min_len = min(len(target_speech), *[len(speech) for speech in background_speeches])
        target_speech = target_speech[:min_len]
        background_speeches = [speech[:min_len] for speech in background_speeches]
    elif mode == 'max':
        max_len = max(len(target_speech), *[len(speech) for speech in background_speeches])
        target_speech = np.pad(target_speech, (0, max_len - len(target_speech)), 'constant')
        background_speeches = [np.pad(speech, (0, max_len - len(speech)), 'constant') for speech in background_speeches]

    # Initialize the mixed speech with the target speech
    mixed_speech = deepcopy(target_speech)

    # Scale and add each background speech based on the given SNR
    # Keep target unchanged, adjust background speech(es) to match target snr_db
    tar_rms = root_mean_square(target_speech)/(10**(snr_db/20))
    # Combine background speeches
    combined_background = np.zeros_like(mixed_speech)
    for speech in background_speeches:
        combined_background += speech
    combined_background_scaled,_ = normalize_rms(combined_background, tar_rms)

    mixed_speech = mixed_speech + combined_background_scaled
    return mixed_speech


def process_and_mix(meta_file, target_sr, mode, mix_dir):
    # meter = pyln.Meter(target_sr)
    meta_data = pd.read_csv(meta_file)
    cnt = 0 
    for i in tqdm(meta_data.index):
        try:
            row = meta_data.loc[i]
            target_data, sr = preprocess_single_audio(row['target_fp'], target_sr, row['target_loudness'])
            background_files = row['background_fps'].split('|')
            background_loudnesses = list(map(float, str(row['background_loudness']).split('|')))
            background_data = [preprocess_single_audio(fp, target_sr, loudness)[0] for fp, loudness in zip(background_files, background_loudnesses)]
            mixed_speech = mix_speeches(target_data, background_data, row['snr_db'], mode)

            audio_clue_data, _ =  preprocess_single_audio(row['audio_clue_fp'], target_sr)

            # prevent clipping
            peak = np.max(np.abs(mixed_speech))
            if peak > 1:
                mixed_speech = mixed_speech / peak
            if max(abs(mixed_speech)) > 1:
                print('warning: clipping!')

            mixed_fp = os.path.join(mix_dir, f'{row["mixed_fp"]}')
            sf.write(mixed_fp, mixed_speech, sr)
            cnt += 1
        except Exception as e:
            print(f"Error processing file {row['target_fp']}: {e}. Skipping this file.")
    print(f"Generated {cnt} / {len(meta_data)} mixtures.")

if __name__ == "__main__":
    splits = ['train', 'test', 'dev'] 

    single_dir = os.path.join(OUTPUT_DIR, f"single_{SAMPLING_FREQUENCY}Hz")
    os.makedirs(single_dir, exist_ok=True)
    
    for split in splits:
        print(f'Now generating mixtures for {split} set...')
        meta_file = os.path.join('metadata', f'{split}_meta.csv')
        mix_dir = os.path.join(OUTPUT_DIR, f'mix_{SAMPLING_FREQUENCY}Hz/{split}_mix')
        os.makedirs(mix_dir, exist_ok=True)
        
        process_and_mix(meta_file, SAMPLING_FREQUENCY, MODE, mix_dir)
