import io
import sys
import httpx
import huggingface_hub
import huggingface_hub.errors
import huggingface_hub.hf_file_system
import huggingface_hub.utils

# Compatibility shims for datasets 5.1.0 with installed huggingface_hub
for attr in ["BucketNotFoundError"]:
    if not hasattr(huggingface_hub.errors, attr):
        setattr(huggingface_hub.errors, attr, getattr(huggingface_hub.errors, "EntryNotFoundError", Exception))

for attr in ["HfFileSystemResolvedBucketPath", "HfFileSystemResolvedRepositoryPath"]:
    if not hasattr(huggingface_hub.hf_file_system, attr):
        setattr(huggingface_hub.hf_file_system, attr, getattr(huggingface_hub.hf_file_system, "HfFileSystemResolvedPath", Exception))

if not hasattr(huggingface_hub.utils, "httpx"):
    huggingface_hub.utils.httpx = httpx

import datasets
import soundfile as sf
from datasets import get_dataset_config_names, load_dataset

token = huggingface_hub.get_token()

configs = get_dataset_config_names("ARTPARK-IISc/Vaani", token=token)
print(f"Total configs: {len(configs)}")

hindi_configs = [c for c in configs if "hindi" in c.lower()]
tamil_configs = [c for c in configs if "tamil" in c.lower()]

print("Hindi matched configs:", hindi_configs)
print("Tamil matched configs:", tamil_configs)

# Inspect a Hindi sample with decode=False
print("\n--- Inspecting Hindi configuration (decode=False) ---")
hindi_cfg = hindi_configs[0]
ds_hindi = load_dataset("ARTPARK-IISc/Vaani", hindi_cfg, split="train", streaming=True, token=token)
ds_hindi = ds_hindi.cast_column("audio", datasets.Audio(decode=False))

hindi_sample = next(iter(ds_hindi))
print("All Schema Keys / Columns:")
for k, v in hindi_sample.items():
    if k == "audio":
        print(f"  audio: dict with keys {list(v.keys())}, path={v.get('path')}, bytes_len={len(v.get('bytes') or b'')} bytes")
    elif k == "image":
        print(f"  image: {type(v)}")
    else:
        print(f"  {k}: {repr(v)}")

# Decode audio bytes with soundfile
audio_bytes = hindi_sample["audio"]["bytes"]
with io.BytesIO(audio_bytes) as bio:
    data, sr = sf.read(bio)
    duration = len(data) / sr
    print(f"\nAudio decoded via soundfile:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Channels: {1 if data.ndim == 1 else data.shape[1]}")
    print(f"  Shape: {data.shape}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Dtype: {data.dtype}")

# Inspect Tamil sample with decode=False
print("\n--- Inspecting Tamil configuration (decode=False) ---")
tamil_cfg = tamil_configs[0]
ds_tamil = load_dataset("ARTPARK-IISc/Vaani", tamil_cfg, split="train", streaming=True, token=token)
ds_tamil = ds_tamil.cast_column("audio", datasets.Audio(decode=False))

tamil_sample = next(iter(ds_tamil))
for k, v in tamil_sample.items():
    if k == "audio":
        print(f"  audio: dict with keys {list(v.keys())}, path={v.get('path')}, bytes_len={len(v.get('bytes') or b'')} bytes")
    elif k == "image":
        print(f"  image: {type(v)}")
    else:
        print(f"  {k}: {repr(v)}")

tamil_audio_bytes = tamil_sample["audio"]["bytes"]
with io.BytesIO(tamil_audio_bytes) as bio:
    t_data, t_sr = sf.read(bio)
    t_dur = len(t_data) / t_sr
    print(f"\nTamil Audio decoded via soundfile:")
    print(f"  Sample Rate: {t_sr} Hz")
    print(f"  Channels: {1 if t_data.ndim == 1 else t_data.shape[1]}")
    print(f"  Duration: {t_dur:.2f} seconds")

# Sample 10 clips from each to get realistic duration and byte size averages
print("\n--- Sampling 10 clips each for accurate size estimation ---")
h_iter = iter(ds_hindi)
h_durations, h_sizes = [], []
h_districts = set()
for _ in range(10):
    item = next(h_iter)
    b = item["audio"]["bytes"]
    with io.BytesIO(b) as bio:
        d, s_rate = sf.read(bio)
        dur = len(d) / s_rate
        h_durations.append(dur)
        h_sizes.append(len(b))
    h_districts.add(item.get("district"))

t_iter = iter(ds_tamil)
t_durations, t_sizes = [], []
t_districts = set()
for _ in range(10):
    item = next(t_iter)
    b = item["audio"]["bytes"]
    with io.BytesIO(b) as bio:
        d, s_rate = sf.read(bio)
        dur = len(d) / s_rate
        t_durations.append(dur)
        t_sizes.append(len(b))
    t_districts.add(item.get("district"))

print(f"Hindi sample districts: {sorted(list(h_districts))}")
print(f"Tamil sample districts: {sorted(list(t_districts))}")
avg_h_dur = sum(h_durations) / len(h_durations)
avg_t_dur = sum(t_durations) / len(t_durations)
avg_h_size_kb = (sum(h_sizes) / len(h_sizes)) / 1024
avg_t_size_kb = (sum(t_sizes) / len(t_sizes)) / 1024

print(f"Hindi average clip duration: {avg_h_dur:.2f}s, average raw file size: {avg_h_size_kb:.1f} KB")
print(f"Tamil average clip duration: {avg_t_dur:.2f}s, average raw file size: {avg_t_size_kb:.1f} KB")

# Estimate size for 100 Hindi + 100 Tamil WAV files
# WAV PCM 16-bit 16kHz mono = 32,000 bytes per second
h_wav_100_mb = (100 * avg_h_dur * 32000) / (1024 * 1024)
t_wav_100_mb = (100 * avg_t_dur * 32000) / (1024 * 1024)
total_mb = h_wav_100_mb + t_wav_100_mb

print(f"\nEstimated local WAV storage size:")
print(f"  100 Hindi WAVs: ~{h_wav_100_mb:.2f} MB")
print(f"  100 Tamil WAVs: ~{t_wav_100_mb:.2f} MB")
print(f"  Total estimated size for 200 WAV files: ~{total_mb:.2f} MB")
