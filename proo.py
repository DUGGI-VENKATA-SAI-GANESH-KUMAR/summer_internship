import os
import numpy as np
import pyaudio
import torch
from plixkws import model, util
import keyboard

stop_key = 'q'
sample_rate = 16000
frames_per_buffer = 512

# Extract word from filenames by removing path and extension
def extract_word_from_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

# Extract word from directory or filename
def extract_word_from_path(filepath, is_dir=False):
    if is_dir:
        return os.path.basename(filepath)
    return os.path.splitext(os.path.basename(filepath))[0]

# Recursively gather audio files and their classes
def gather_audio_files(base_folder):
    audio_files = []
    classes = []
    for root, _, files in os.walk(base_folder):
        if files:
            keyword = extract_word_from_path(root, is_dir=True)
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
                    classes.append(f"{keyword}-{extract_word_from_path(file)}")
    return audio_files, classes

# Existing support examples
support_examples = ["./test_clips/no-keyword.wav"]
classes = [extract_word_from_filename(f) for f in support_examples]
int_indices = list(range(len(classes)))

# New Urdu dataset
urdu_folder = r'C:\Users\duggi\OneDrive\Documents\Project\plix\test_clips\urdu_clips_robust'
urdu_files, urdu_classes = gather_audio_files(urdu_folder)
urdu_indices = list(range(len(classes), len(classes) + len(urdu_files)))

# New English dataset
english_folder = r'C:\Users\duggi\OneDrive\Documents\Project\plix\test_clips\english_clips_robust'
english_files, english_classes = gather_audio_files(english_folder)
english_indices = list(range(len(classes) + len(urdu_files), len(classes) + len(urdu_files) + len(english_files)))

# Combine all datasets
support_examples.extend(urdu_files)
support_examples.extend(english_files)
classes.extend(urdu_classes)
classes.extend(english_classes)
int_indices.extend(urdu_indices)
int_indices.extend(english_indices)

support = {
    "paths": support_examples,
    "classes": classes,
    "labels": torch.tensor(int_indices)
}

# Load and batch the audio clips
support["audio"] = torch.stack([util.load_clip(path) for path in support["paths"]])
support = util.batch_device(support, device="cpu")

fws_model = model.load(encoder_name="small", language="multi", device="cpu")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=frames_per_buffer)

frames = []
while True:
    data = stream.read(frames_per_buffer)
    buffer = np.frombuffer(data, dtype=np.int16)
    frames.append(buffer)
    if len(frames) * frames_per_buffer / sample_rate >= 1:
        audio = np.concatenate(frames)
        audio = audio.astype(float) / np.iinfo(np.int16).max
        query = {"audio": torch.tensor(audio[np.newaxis, np.newaxis, :], dtype=torch.float32)}
        query = util.batch_device(query, device="cpu")
        with torch.no_grad():
            predictions, distances = fws_model(support, query)
            predicted_class = classes[predictions.item()]
            distance = distances.min().item()
            if distance<20:
                print(f"Predicted class: {predicted_class}, Distance: {distance}")
            else:
                continue
        frames = []

    if keyboard.is_pressed(stop_key):
        print("Key pressed! Stopping simulation.")
        break

stream.stop_stream()
stream.close()
p.terminate()
