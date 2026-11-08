# Vaani Dataset Offline Verification Report

## Hindi Audio & Metadata Verification

- **Metadata records found**: 100
- ✅ **All required CSV columns present**: `clip_id, language, duration, transcript, speakerID, state, district, source_configuration, original_sample_rate, audio_file_path`
- **Files on disk**: 100 files found in `data/vaani/hindi/audio`
- ✅ **File Count**: Exactly 100 WAV files exist.
- ✅ **Corrupted files**: 0
- ✅ **Empty files**: 0
- **Total Hindi storage**: 26.58 MB (27,866,416 bytes)

### Sample Audio Read Test (First 3 clips)
| File | Sample Rate | Channels | Duration | Dtype | Subtype | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hindi_001.wav` | 16000 Hz | 1 | 6.32s | float64 | PCM_16 | ✅ Valid |
| `hindi_002.wav` | 16000 Hz | 1 | 12.44s | float64 | PCM_16 | ✅ Valid |
| `hindi_003.wav` | 16000 Hz | 1 | 8.14s | float64 | PCM_16 | ✅ Valid |

---

## Tamil Audio & Metadata Verification

- **Metadata records found**: 100
- ✅ **All required CSV columns present**: `clip_id, language, duration, transcript, speakerID, state, district, source_configuration, original_sample_rate, audio_file_path`
- **Files on disk**: 100 files found in `data/vaani/tamil/audio`
- ✅ **File Count**: Exactly 100 WAV files exist.
- ✅ **Corrupted files**: 0
- ✅ **Empty files**: 0
- **Total Tamil storage**: 18.71 MB (19,619,986 bytes)

### Sample Audio Read Test (First 3 clips)
| File | Sample Rate | Channels | Duration | Dtype | Subtype | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tamil_001.wav` | 16000 Hz | 1 | 6.77s | float64 | PCM_16 | ✅ Valid |
| `tamil_002.wav` | 16000 Hz | 1 | 13.26s | float64 | PCM_16 | ✅ Valid |
| `tamil_003.wav` | 16000 Hz | 1 | 8.33s | float64 | PCM_16 | ✅ Valid |

---

## Overall Summary

- **Total Local Audio Files**: 200 / 200
- **Total Disk Space Consumed**: **45.29 MB** (47,486,402 bytes)
- **Offline Usability**: All files verified with `soundfile` without any active network connection required.

### Final Status: **✅ ALL CHECKS PASSED (100% Offline Ready)**
