import numpy as np
import librosa
from sklearn.decomposition import FastICA
import soundfile as sf
import sounddevice as sd

# Load audio files
signal1, samp_rate1 = librosa.load('audio files\\music_as_high.wav')
signal2, samp_rate2 = librosa.load('audio files\\speech_as_high.wav')

# Ensure both signals have the same length
length_of_shorter = min(len(signal1), len(signal2))
signal1 = signal1[:length_of_shorter]
signal2 = signal2[:length_of_shorter]

# Combine the audio files into one matrix (required for FastICA function)
signal_comb = np.column_stack((signal1, signal2))

# Use FastICA function to separate the signals into the 2 original sources
ica = FastICA(n_components=2, random_state=0)
separated = ica.fit_transform(signal_comb)

# Normalize the separated signals
separated_1 = separated[:, 0] / np.max(np.abs(separated[:, 0]))
separated_2 = separated[:, 1] / np.max(np.abs(separated[:, 1]))

# Play the separated audio files
sd.play(separated_1, samp_rate1)
sd.play(separated_2, samp_rate2)

# Wait until playback is finished
sd.wait()


# Save the separated audio signals as WAV files
sf.write('separated_music.wav', separated_1, samp_rate1)
sf.write('separated_speech.wav', separated_2, samp_rate2)

