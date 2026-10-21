#!/usr/bin/env python3
"""
generate_dataset.py
-------------------
Pembangkitan dataset spike EKG spatio-temporal [T=64, C=30] menggunakan Direct Rate Spike Encoding
berdasarkan Sulistiyanto et al. (MCSoC 2025) & Asghar et al. (Sensors 2021).

Sumber Data:
- SNN4FPGA/Encoding And Train/dataset_ecg/{N, V, ...}
- Kelas 0: Normal Beat ('N')
- Kelas 1: Premature Ventricular Contraction / Ventricular Arrhythmia ('V')

Keluaran:
- ecg_spikes_dataset.pt (Dataset PyTorch untuk pelatihan SNN)
- dataset_summary.json (Metadata statistik)
"""

import os
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch

# 30 titik anchor (P: 15, QRS: 9, T: 6) dari direct_spike_encoding
ANCHORS = [
    18, 22, 26, 30, 33, 37, 41, 45, 49, 53, 57, 60, 64, 68, 72,   # P region (15)
    78, 81, 84, 87, 90, 93, 96, 99, 102,                          # QRS region (9)
    112, 132, 153, 174, 195, 216                                  # T region (6)
]
NUM_ANCHORS = 30
NUM_SLOTS = 64
NUM_SAMPLES = 234

def abs_s16(v: int) -> int:
    """Signed 16-bit absolute value replication to match RTL."""
    a = -v if v < 0 else v
    if not -32768 <= a <= 32767:
        a = ((a + 0x8000) & 0xFFFF) - 0x8000
    return a

def direct_rate_encode(samples_int16: np.ndarray, anchors=ANCHORS, num_slots=NUM_SLOTS) -> np.ndarray:
    """
    Direct Rate Spike Encoding (Bresenham DDA algorithm).
    Menghasilkan raster spike biner [64, 30].
    """
    # 1. Cari peak amplitude detak
    peak = 1
    for v in samples_int16:
        a = abs_s16(int(v))
        if a > peak:
            peak = a
            
    # 2. Hitung rate k[c] per channel anchor
    k_rate = []
    for c in anchors:
        amp = abs(int(samples_int16[c])) & 0xFFFF
        k = ((amp * num_slots) + (peak >> 1)) // peak
        k_rate.append(k & 0x7F)
        
    # 3. Bresenham / DDA spike generator
    raster = np.zeros((num_slots, len(anchors)), dtype=np.float32)
    acc = [0] * len(anchors)
    for s in range(num_slots):
        for c in range(len(anchors)):
            if acc[c] + k_rate[c] >= num_slots:
                raster[s, c] = 1.0
                acc[c] = (acc[c] + k_rate[c] - num_slots) & 0xFF
            else:
                acc[c] = (acc[c] + k_rate[c]) & 0xFF
                
    return raster

SCRIPT_DIR = Path(__file__).resolve().parent

def load_or_create_ecg_spikes(data_dir=None, output_path=None, test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if output_path is None:
        output_path = SCRIPT_DIR / "ecg_spikes_dataset.pt"
    else:
        output_path = Path(output_path)

    # Lokasi dataset default
    if data_dir is None:
        candidates = [
            Path("../SNN4FPGA/Encoding And Train/dataset_ecg"),
            Path("../../SNN4FPGA/Encoding And Train/dataset_ecg"),
            Path("dataset_ecg"),
            Path("C:/Users/user/Documents/SKRIPSII/SNN4FPGA/Encoding And Train/dataset_ecg")
        ]
        for c in candidates:
            if c.is_dir():
                data_dir = c
                break

    print("=" * 65)
    print(" 1. GENERATE ECG SPIKE DATASET (DIRECT RATE ENCODING [64, 30])")
    print("=" * 65)

    all_samples = []
    all_labels = []

    if data_dir is not None and Path(data_dir).is_dir():
        print(f"[*] Membaca data ECG riil dari: {Path(data_dir).resolve()}")
        # Ambil sampel kelas N (label 0: Normal) dan V (label 1: Ventricular ectopic / Arrhythmia)
        # Tugas klasifikasi biner untuk 1 LIF neuron
        cls_map = {"N": 0, "V": 1}
        for cls_name, label in cls_map.items():
            folder = Path(data_dir) / cls_name
            files = sorted(glob.glob(str(folder / "*.npz")))
            print(f"    - Kelas '{cls_name}' (Label {label}): Ditemukan {len(files)} file .npz")
            for f in files:
                data = np.load(f, allow_pickle=True)
                sig = data["signal"] # float32 mV
                # Kuantisasi ke Q2.13 int16 persis seperti pipeline FPGA
                sig_int16 = np.clip(np.round(sig * 8192.0), -32768, 32767).astype(np.int16)
                # Direct Rate Spike Encoding [64, 30]
                raster = direct_rate_encode(sig_int16)
                all_samples.append(raster)
                all_labels.append(label)
    else:
        print("[!] Direktori dataset ECG tidak ditemukan, membangkitkan data sintetis fisiologis...")
        # Fallback pembangkitan sintetis jika dataset asli tidak ada
        for i in range(300):
            # Normal beat (N)
            sig_n = np.random.normal(0, 100, 234).astype(np.int16)
            raster_n = direct_rate_encode(sig_n)
            all_samples.append(raster_n)
            all_labels.append(0)
            # Arrhythmia beat (V)
            sig_v = np.random.normal(0, 500, 234).astype(np.int16)
            raster_v = direct_rate_encode(sig_v)
            all_samples.append(raster_v)
            all_labels.append(1)

    all_samples = np.array(all_samples, dtype=np.float32) # [N, 64, 30]
    all_labels = np.array(all_labels, dtype=np.int64)      # [N]

    print(f"[*] Total sampel ter-encode: {all_samples.shape[0]}")
    print(f"[*] Dimensi matriks per sampel: {all_samples.shape[1:]} [T=64 time-slots, C=30 channels]")
    print(f"    - Label 0 (Normal): {(all_labels == 0).sum()} sampel")
    print(f"    - Label 1 (Arrhythmia): {(all_labels == 1).sum()} sampel")

    # Shuffle dan train/test split
    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1.0 - test_ratio))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    x_train = torch.tensor(all_samples[train_idx], dtype=torch.float32)
    y_train = torch.tensor(all_labels[train_idx], dtype=torch.long)
    x_test = torch.tensor(all_samples[test_idx], dtype=torch.float32)
    y_test = torch.tensor(all_labels[test_idx], dtype=torch.long)

    # Simpan dataset PyTorch
    torch.save({
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "num_anchors": NUM_ANCHORS,
        "num_slots": NUM_SLOTS,
        "anchors": ANCHORS
    }, output_path)

    print(f"[+] Dataset berhasil disimpan ke: {Path(output_path).resolve()}")
    print(f"    - Train set: {x_train.shape[0]} sampel")
    print(f"    - Test set : {x_test.shape[0]} sampel")

    # Simpan metadata ringkasan JSON
    summary = {
        "total_samples": int(len(all_labels)),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "time_steps": NUM_SLOTS,
        "num_inputs": NUM_ANCHORS,
        "normal_count": int((all_labels == 0).sum()),
        "arrhythmia_count": int((all_labels == 1).sum())
    }
    summary_path = output_path.parent / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return output_path

if __name__ == "__main__":
    load_or_create_ecg_spikes()
