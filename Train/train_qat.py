#!/usr/bin/env python3
"""
train_qat.py
------------
Pelatihan Model 1 Neuron Digital LIF (snnTorch + PyTorch)
Menerapkan:
1. Pelatihan Warmup Float32 (15 Epoch)
2. Quantization-Aware Training (QAT) Format Q4.4 (8-bit signed) via Straight-Through Estimator (STE)
3. Soft Reset (subtract) & Surrogate Gradient (atan)
4. Evaluasi Hardware-Friendly (Spike Rate Thresholding)

Output:
- best_lif_model.pt
- training_history.png
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import snntorch as snn
from snntorch import surrogate

SCRIPT_DIR = Path(__file__).resolve().parent

# ==============================================================================
# 1. Straight-Through Estimator (STE) untuk Kuantisasi Q4.4
# ==============================================================================
class STEQuantizeQ44(torch.autograd.Function):
    """
    Format Q4.4 (Fixed-Point 8-bit signed):
    - 1 Sign bit, 3 Integer bits, 4 Fractional bits
    - Skala: 2^4 = 16.0
    - Rentang Integer: [-128, 127] -> Rentang Float: [-8.0, 7.9375]
    """
    @staticmethod
    def forward(ctx, w):
        scaled = w * 16.0
        clamped_int = torch.clamp(torch.round(scaled), -128.0, 127.0)
        return clamped_int / 16.0

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Gradien dilewatkan secara identik tanpa perubahan
        return grad_output

def quantize_q44(w):
    return STEQuantizeQ44.apply(w)

# ==============================================================================
# 2. Arsitektur Single LIF Neuron (Sesuai README & Sensors 2021)
# ==============================================================================
class SingleLIF(nn.Module):
    def __init__(self, num_inputs=30, beta=0.875, threshold=1.0):
        super().__init__()
        self.num_inputs = num_inputs
        self.beta = beta
        self.threshold = threshold
        
        # 1 Lapisan Sinaps Linier tanpa bias (0 DSP, murni AND-gating di FPGA)
        self.fc = nn.Linear(num_inputs, 1, bias=False)
        
        # Inisialisasi bobot seragam kecil
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)

        # Neuron Leaky Integrate-and-Fire dengan Soft Reset
        # Beta = 0.875 (1 - 2^-3) mencerminkan Shift-and-Subtract (DECAY_SHIFT = 3)
        self.lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=surrogate.atan(),
            reset_mechanism="subtract"
        )

    def forward(self, x_spikes, use_qat=False):
        """
        x_spikes: Tensor [Batch, TimeSteps=64, Inputs=30]
        Return: (spk_out [Batch, TimeSteps, 1], mem_out [Batch, TimeSteps, 1])
        """
        batch_size = x_spikes.size(0)
        time_steps = x_spikes.size(1)
        
        # Bobot sinaps (kuantisasi Q4.4 jika mode QAT)
        w = quantize_q44(self.fc.weight) if use_qat else self.fc.weight
        
        mem = torch.zeros(batch_size, 1, device=x_spikes.device)
        spk_list = []
        mem_list = []
        
        for t in range(time_steps):
            # Synaptic current: input spike biner * bobot
            cur = torch.matmul(x_spikes[:, t], w.t())
            spk, mem = self.lif(cur, mem)
            spk_list.append(spk)
            mem_list.append(mem)
            
        spk_out = torch.stack(spk_list, dim=1) # [B, T, 1]
        mem_out = torch.stack(mem_list, dim=1) # [B, T, 1]
        return spk_out, mem_out

# ==============================================================================
# 3. Fungsi Pelatihan & Evaluasi
# ==============================================================================
def train_model(
    dataset_path=None,
    warmup_epochs=15,
    qat_epochs=20,
    batch_size=32,
    lr=0.01,
    device="cpu"
):
    print("=" * 65)
    print(" 2. TRAINING SINGLE LIF NEURON DENGAN QAT (FORMAT Q4.4)")
    print("=" * 65)
    
    if dataset_path is None:
        dataset_path = SCRIPT_DIR / "ecg_spikes_dataset.pt"
    
    if not Path(dataset_path).is_file():
        # Coba generate dataset jika belum ada
        from generate_dataset import load_or_create_ecg_spikes
        load_or_create_ecg_spikes(output_path=dataset_path)

    data = torch.load(dataset_path, weights_only=True)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    num_inputs = data.get("num_anchors", 30)
    model = SingleLIF(num_inputs=num_inputs, beta=0.875, threshold=1.0).to(device)

    # Optimizer Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs + qat_epochs)

    # Target Spike Count:
    # Kelas 0 (Normal) -> 0 - 2 spikes
    # Kelas 1 (Arrhythmia) -> 16 - 25 spikes
    target_low = 1.0
    target_high = 20.0

    history = {
        "epoch": [], "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [], "mode": []
    }

    best_test_acc = 0.0
    best_weights = None

    total_epochs = warmup_epochs + qat_epochs
    print(f"[*] Total Epochs: {total_epochs} ({warmup_epochs} Float32 Warmup + {qat_epochs} Q4.4 QAT)")
    print(f"[*] Perangkat komputasi: {device}")

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        is_qat = (epoch > warmup_epochs)
        mode_str = "QAT Q4.4" if is_qat else "Float32"
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            spk_out, _ = model(x_b, use_qat=is_qat)
            spike_counts = spk_out.sum(dim=1).squeeze(-1) # [B]

            # Target spike count berdasarkan label
            target_counts = torch.where(y_b == 1, target_high, target_low).float()

            # MSE Loss pada spike count
            loss = F.mse_loss(spike_counts, target_counts)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_b.size(0)

            # Prediksi berdasarkan ambang batas spike (misal threshold = (target_low + target_high)/2)
            preds = (spike_counts >= ((target_low + target_high) / 2.0)).long()
            correct_train += (preds == y_b).sum().item()
            total_train += x_b.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Evaluasi pada Test Set
        model.eval()
        test_loss_sum = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                spk_out, _ = model(x_b, use_qat=is_qat)
                spike_counts = spk_out.sum(dim=1).squeeze(-1)
                target_counts = torch.where(y_b == 1, target_high, target_low).float()
                
                loss = F.mse_loss(spike_counts, target_counts)
                test_loss_sum += loss.item() * x_b.size(0)

                preds = (spike_counts >= ((target_low + target_high) / 2.0)).long()
                correct_test += (preds == y_b).sum().item()
                total_test += x_b.size(0)

        test_loss = test_loss_sum / total_test
        test_acc = (correct_test / total_test) * 100.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["mode"].append(mode_str)

        if epoch % 5 == 0 or epoch == 1 or epoch == (warmup_epochs + 1) or epoch == total_epochs:
            print(f"Epoch [{epoch:02d}/{total_epochs:02d}] ({mode_str:8s}) | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}%")

        # Simpan checkpoint model terbaik selama fase QAT
        if is_qat and test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_weights = model.fc.weight.detach().clone()

    elapsed = time.time() - start_time
    print(f"\n[+] Pelatihan selesai dalam {elapsed:.2f} detik.")
    print(f"[+] Akurasi Uji Terbaik pada QAT Q4.4: {best_test_acc:.2f}%")

    # Muat bobot terbaik dan simpan
    if best_weights is not None:
        model.fc.weight.data.copy_(best_weights)

    save_model_path = SCRIPT_DIR / "best_lif_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "weight_float": model.fc.weight.detach().cpu(),
        "weight_q44_int": torch.clamp(torch.round(model.fc.weight.detach().cpu() * 16.0), -128, 127).to(torch.int8),
        "beta": model.beta,
        "threshold": model.threshold,
        "best_test_acc": best_test_acc,
        "num_inputs": num_inputs
    }, save_model_path)
    print(f"[+] Model terbaik disimpan ke: {save_model_path.resolve()}")

    # Visualisasi Kurva Pelatihan
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss", color="crimson")
    plt.axvline(x=warmup_epochs, color="gray", linestyle="--", label="Start QAT")
    plt.title("Kurva Loss (Warmup & QAT)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc", color="royalblue")
    plt.plot(history["epoch"], history["test_acc"], label="Test Acc", color="crimson")
    plt.axvline(x=warmup_epochs, color="gray", linestyle="--", label="Start QAT")
    plt.title(f"Akurasi Klasifikasi (Best: {best_test_acc:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = SCRIPT_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] Grafik kurva pelatihan disimpan ke: {plot_path.resolve()}")

    return model

if __name__ == "__main__":
    train_model()

