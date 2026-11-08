import json, sys

for clip in ['hindi_001', 'hindi_002', 'hindi_003']:
    d = json.load(open(f'outputs/json/{clip}_analysis.json'))
    vp = d['global_info']['voice_properties']
    sp = d['global_info']['signal_properties']
    rp = d['global_info']['recording_properties']
    print(f'--- {clip} ---')
    print(f'  Duration: {d["metadata"]["duration_seconds"]}s | Channels: {d["metadata"]["num_channels"]}')
    print(f'  Peak: {d["metadata"]["peak_amplitude"]["dbfs"]} dBFS | RMS: {d["metadata"]["rms_amplitude"]["dbfs"]} dBFS')
    print(f'  Clipping: {d["metadata"]["clipping_percentage"]}%  Silence: {d["metadata"]["silence_percentage"]}%')
    print(f'  Spectral centroid: {sp["spectral_centroid"]["mean_hz"]} Hz')
    print(f'  Voiced%: {vp["voiced_percentage"]}%')
    print(f'  Median F0: {vp["median_f0_hz"]} Hz  Range: {vp["min_reliable_f0_hz"]}-{vp["max_reliable_f0_hz"]} Hz')
    print(f'  HNR: mean={vp["harmonic_to_noise_ratio_db"]["mean_db"]} dB')
    print(f'  Speech rate: {vp["estimated_speech_rate"]["syllables_per_second"]} syl/s')
    print(f'  SNR: {rp["estimated_snr_db"]["value"]} dB | Noise floor: {rp["background_noise_floor_dbfs"]["value"]} dBFS')
    print(f'  Speech segs: {len(d["speech_segments"])}  Local events: {len(d["local_events"])}  Acoustic changes: {len(d["acoustic_changes"])}  Timeline items: {len(d["timeline"])}')
    if d['local_events']:
        print(f'  Events: {[(e["event_label"], e["start_time"], e["end_time"]) for e in d["local_events"][:5]]}')
    if d['acoustic_changes']:
        print(f'  Changes (first 3): {[(c["event_label"], c["start_time"], c["confidence"]) for c in d["acoustic_changes"][:3]]}')
    print()

