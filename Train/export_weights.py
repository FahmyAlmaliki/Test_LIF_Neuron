#!/usr/bin/env python3
"""
export_weights.py
-----------------
Fase 2: Ekspor Bobot & Parameter ke Format Hardware (FPGA Vivado)
1. Membaca model hasil QAT (best_lif_model.pt)
2. Mengonversi bobot Float32 ke Fixed-Point 8-bit bertanda Q4.4:
   Weight_fixed = clamp(round(Weight_float * 16), -128, 127)
3. Menyimpan bobot ke 'FPGA/weight_q44.mem' (format heksadesimal $readmemh)
4. Mengekspor vektor uji (stimulus) 'FPGA/input_spikes.mem', 'golden_spikes.mem', dan 'golden_vmem.mem'
   untuk verifikasi ekuivalensi hardware-software bit-exact.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FPGA_DIR = PROJECT_DIR / "FPGA"

def to_hex_2s_comp_8bit(val: int) -> str:
    """Konversi integer bertanda 8-bit [-128, 127] ke format heksadesimal 2 digit."""
    v = int(val) & 0xFF
    return f"{v:02X}"

def to_hex_2s_comp_16bit(val: int) -> str:
    """Konversi integer bertanda 16-bit [-32768, 32767] ke format heksadesimal 4 digit."""
    v = int(val) & 0xFFFF
    return f"{v:04X}"

def simulate_hardware_bitexact(x_spikes_sample: np.ndarray, weights_int8: np.ndarray, decay_shift=3, v_th=16):
    """
    Simulasi logika hardware PERSIS sesuai register-transfer level (RTL):
    - x_spikes_sample: [T=64, C=30]
    - weights_int8: [30]
    - decay_shift = 3 (Shift-and-Subtract: v_mem - (v_mem >> 3))
    - v_th = 16 (Q.4 equivalent of 1.0)
    """
    num_slots = x_spikes_sample.shape[0]
    num_channels = x_spikes_sample.shape[1]
    
    vmem = 0
    spike_out_seq = []
    vmem_seq = []
    
    for t in range(num_slots):
        # 1. Synaptic Adder Tree (AND-gating + sum)
        acc_val = 0
        for c in range(num_channels):
            if x_spikes_sample[t, c] > 0.5:
                acc_val += int(weights_int8[c])
                
        # 2. Leakage Unit (Arithmetic Right Shift)
        # Python '>>' untuk integer negatif berperilaku seperti arithmetic shift (>>> di Verilog)
        v_decay = vmem - (vmem >> decay_shift)
        
        # 3. Summing & Threshold Comparison
        v_sum = v_decay + acc_val
        spk = 1 if (v_sum >= v_th) else 0
        
        # 4. Soft Reset (v_next = v_sum - v_th jika spike, else v_sum)
        v_next = (v_sum - v_th) if (spk == 1) else v_sum
        
        spike_out_seq.append(spk)
        vmem_seq.append(v_next)
        vmem = v_next

    return np.array(spike_out_seq, dtype=np.int32), np.array(vmem_seq, dtype=np.int32)

def export_all():
    print("=" * 65)
    print(" 3. EKSPOR BOBOT & VEKTOR UJI KE FORMAT HARDWARE (FPGA)")
    print("=" * 65)

    model_path = SCRIPT_DIR / "best_lif_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {model_path}. Jalankan train_qat.py terlebih dahulu.")

    os.makedirs(FPGA_DIR, exist_ok=True)

    # 1. Baca Bobot Terlatih
    checkpoint = torch.load(model_path, weights_only=True)
    w_float = checkpoint["weight_float"].squeeze().numpy() # [30]
    
    # Skalakan ke Q4.4 signed 8-bit
    w_scaled = np.round(w_float * 16.0)
    w_int = np.clip(w_scaled, -128, 127).astype(np.int8)

    print(f"[*] Jumlah bobot sinaps: {len(w_int)}")
    print(f"[*] Format: Q4.4 Signed Fixed-Point (Skala = 16.0, Rentang = [-128, 127])")

    # 2. Tulis weight_q44.mem untuk Vivado ($readmemh)
    weight_mem_path = FPGA_DIR / "weight_q44.mem"
    with open(weight_mem_path, "w") as f:
        f.write("// ================================================================\n")
        f.write("// weight_q44.mem - Bobot Sinaps 8-Bit Q4.4 untuk Digital LIF Neuron\n")
        f.write("// Baris i = Bobot Kanal i (Format Hexadecimal Two's Complement)\n")
        f.write("// ================================================================\n")
        for i, (wf, wi) in enumerate(zip(w_float, w_int)):
            h = to_hex_2s_comp_8bit(wi)
            f.write(f"{h} // W[{i:02d}]: Float={wf:7.4f} -> Int={wi:4d} -> 0x{h}\n")
    print(f"[+] File bobot hardware berhasil diekspor: {weight_mem_path.resolve()}")

    # 3. Pilih Sampel Uji untuk Verifikasi Testbench
    dataset_path = SCRIPT_DIR / "ecg_spikes_dataset.pt"
    test_spikes = None
    sample_label = 1
    if dataset_path.is_file():
        ds = torch.load(dataset_path, weights_only=True)
        x_test = ds["x_test"].numpy()
        y_test = ds["y_test"].numpy()
        # Ambil sampel Arrhythmia (Label 1) pertama di test set agar menghasilkan lonjakan spike
        arrhythmia_indices = np.where(y_test == 1)[0]
        chosen_idx = arrhythmia_indices[0] if len(arrhythmia_indices) > 0 else 0
        test_spikes = x_test[chosen_idx]
        sample_label = int(y_test[chosen_idx])
        print(f"[*] Menggunakan sampel uji index #{chosen_idx} (Label = {sample_label})")
    else:
        test_spikes = (np.random.rand(64, 30) > 0.8).astype(np.float32)

    # 4. Ekspor input_spikes.mem (64 baris x 30-bit kata per baris)
    input_mem_path = FPGA_DIR / "input_spikes.mem"
    with open(input_mem_path, "w") as f:
        f.write("// ================================================================\n")
        f.write("// input_spikes.mem - Stimulus Spike Input [T=64 time-slots, C=30]\n")
        f.write("// Format: 30-bit Hexadecimal per slot (Bit i = Spike_In[i])\n")
        f.write("// ================================================================\n")
        for t in range(64):
            word30 = 0
            for c in range(30):
                if test_spikes[t, c] > 0.5:
                    word30 |= (1 << c)
            f.write(f"{word30:08X} // Slot {t:02d}: 30'h{word30:08X}\n")
    print(f"[+] File stimulus input berhasil diekspor: {input_mem_path.resolve()}")

    # 5. Hitung Golden Outputs Hardware
    golden_spikes, golden_vmem = simulate_hardware_bitexact(test_spikes, w_int, decay_shift=3, v_th=16)

    # 6. Ekspor golden_spikes.mem
    golden_spk_path = FPGA_DIR / "golden_spikes.mem"
    with open(golden_spk_path, "w") as f:
        f.write("// ================================================================\n")
        f.write("// golden_spikes.mem - Golden Spike Out (1-bit binary per time-slot)\n")
        f.write("// ================================================================\n")
        for t, spk in enumerate(golden_spikes):
            f.write(f"{spk} // Slot {t:02d}: Spike={spk}\n")
    print(f"[+] File golden spikes berhasil diekspor: {golden_spk_path.resolve()}")

    # 7. Ekspor golden_vmem.mem
    golden_vmem_path = FPGA_DIR / "golden_vmem.mem"
    with open(golden_vmem_path, "w") as f:
        f.write("// ================================================================\n")
        f.write("// golden_vmem.mem - Golden Vmem 16-Bit Q12.4 (Hex Two's Comp per slot)\n")
        f.write("// ================================================================\n")
        for t, vm in enumerate(golden_vmem):
            h = to_hex_2s_comp_16bit(vm)
            f.write(f"{h} // Slot {t:02d}: Vmem={vm:5d} ({vm/16.0:6.3f}V) -> 0x{h}\n")
    print(f"[+] File golden vmem berhasil diekspor: {golden_vmem_path.resolve()}")

    print("\nRingkasan Stimulus Verifikasi:")
    print(f"  - Total Spike Input Terdeteksi  : {int(test_spikes.sum())} spikes")
    print(f"  - Total Spike Output Neuron (SNN) : {int(golden_spikes.sum())} spikes")
    print(f"  - Vmem Minimum / Maksimum       : {golden_vmem.min()} ({golden_vmem.min()/16.0:.2f}V) / {golden_vmem.max()} ({golden_vmem.max()/16.0:.2f}V)")
    print("=" * 65)

if __name__ == "__main__":
    export_all()

