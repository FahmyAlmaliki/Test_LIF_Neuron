# Implementation & Testing Roadmap: Single Digital LIF Neuron Hardware Design

Repositori dan panduan ini berisi arsitektur rinci **1 Neuron Digital Leaky Integrate-and-Fire (LIF)** serta alur langkah demi langkah (*step-by-step roadmap*) dari pelatihan *software* hingga pengujian *hardware* FPGA di Vivado.

---

## 1. Arsitektur Hardware Digital 1 Neuron LIF

Arsitektur hardware digital murni (*all-digital*) untuk 1 neuron LIF ini dirancang berdasarkan prinsip **Sensors 2021 (Asghar et al.)** dan kerangka kerja **SNN4FPGA**. Arsitektur ini menghindari penggunaan komponen analog maupun unit perkalian *floating-point* berat, sehingga sangat hemat area (LUT/Flip-Flop) dan daya pada FPGA.

### Diagram Blok Rangkaian Hardware Digital
```text
                                +-------------------------------------------------------+
                                |                 NEURON LIF DIGITAL                    |
  Spike_In[0] --[Mux / AND]----+|                                                       |
                 (0 / W[0])     |                                                       |
  Spike_In[1] --[Mux / AND]----+|   +--------------+   +------------+   +-------------+ |
                 (0 / W[1])     +-->| Akumulator   |-->| Membrane   |-->| Komparator  |+---> Spike_Out
  ...             ...          +|   | (Adder Tree) |   | Register   |   | (Vmem>=Vth) | |        |
  Spike_In[N] --[Mux / AND]----+|   +--------------+   +------------+   +-------------+ |        |
                 (0 / W[N])     |          ^                 |                 |        |        |
                                |          |          [Leakage Unit]           |        |        |
                                |          |         (Shift-Subtract)          v        |        |
                                |          +----------------------------- [Reset Mux] <-+--------+
                                +-------------------------------------------------------+
```

### Rincian Komponen Utama:
1. **Sinaps / Selektor Input (Mux / AND-Gating)**:
   * Mengalikan *spike* biner masukan (`Spike_In` $\in \{0, 1\}$) dengan bobot sinaps ($W_i$).
   * Menggunakan gerbang logika **AND** atau **Multiplexer**: Jika `Spike_In` = 1, nilai bobot terkuantisasi $W_i$ diteruskan. Jika `Spike_In` = 0, nilai 0 diteruskan (0 DSP slice).
2. **Akumulator / Pohon Penjumlah (*Adder Tree*)**:
   * Menjumlahkan seluruh luaran sinaps aktif secara sekuensial atau paralel ke dalam akumulator (`acc_reg`).
3. **Register Potensial Membran (*Membrane Register*)**:
   * Register biner (misalnya 24-bit atau 32-bit `signed integer`) yang menyimpan nilai potensial membran ($V_{mem}$).
4. **Unit Peluruhan (*Leakage Unit*)**:
   * Mengimplementasikan sifat kebocoran eksponensial ($\beta$) menggunakan operasi aritmatika digital **Shift-and-Subtract**:
     $$\text{V}_{decay} = \text{V}_{mem} - (\text{V}_{mem} \gg \text{DECAY\_SHIFT})$$
5. **Komparator Ambang Batas (*Threshold Comparator*)**:
   * Membandingkan $V_{sum} = V_{decay} + acc\_reg$ dengan nilai ambang batas digital ($V_{th}$).
   * Jika $V_{sum} \ge V_{th}$, menghasilkan keluaran biner `Spike_Out = 1`.
6. **Mekanisme Reset (*Soft Reset / Hard Reset*)**:
   * Saat `Spike_Out = 1`, potensial membran di-reset menggunakan metode *soft reset* ($\text{V}_{new} = \text{V}_{sum} - \text{V}_{th}$) atau *hard reset* ($\text{V}_{new} = 0$).

---

## 2. Persamaan Logika Hardware Digital

$$\text{V}_{decay}[n] = \text{V}_{mem}[n] - (\text{V}_{mem}[n] \gg \text{DECAY\_SHIFT})$$

$$\text{V}_{sum}[n] = \text{V}_{decay}[n] + \sum_{i=0}^{N-1} (W_i \cdot \text{Spike\_In}_i[n])$$

$$\text{Spike\_Out}[n] = \begin{cases} 1, & \text{jika } \text{V}_{sum}[n] \ge \text{V}_{th} \\ 0, & \text{jika } \text{V}_{sum}[n] < \text{V}_{th} \end{cases}$$

$$\text{V}_{mem}[n+1] = \begin{cases} \text{V}_{sum}[n] - \text{V}_{th}, & \text{jika } \text{Spike\_Out}[n] = 1 \text{ (Soft Reset)} \\ \text{V}_{sum}[n], & \text{jika } \text{Spike\_Out}[n] = 0 \end{cases}$$

---

## 3. Langkah-Langkah Perancangan & Pengujian (*Step-by-Step Roadmap*)

```text
+------------------------+      +------------------------+      +------------------------+      +------------------------+
| Phase 1: Software Train| ---> | Phase 2: Export Weight | ---> | Phase 3: Verilog RTL   | ---> | Phase 4: Vivado Sim    |
| (PyTorch + snnTorch)   |      | (Q4.4 Fixed-Point .mem)|      | (Single LIF Module)    |      | (Testbench Verification|
+------------------------+      +------------------------+      +------------------------+      +------------------------+
```

### **Fase 1: Pelatihan Software & Quantization-Aware Training (QAT)**
1. **Persiapan Data Spike EKG**: Konversi data EKG ke dalam bentuk matriks *spike* biner spatio-temporal $[T = 64, C = 30]$ menggunakan *Direct Rate Encoding* (1 kHz/mV).
2. **Inisialisasi Model snnTorch**: Buat model 1 neuron LIF di PyTorch dengan mekanisme *soft reset* dan *surrogate gradient* (`atan`):
   ```python
   import torch
   import snntorch as snn
   from snntorch import surrogate

   class SingleLIF(torch.nn.Module):
       def __init__(self, num_inputs):
           super().__init__()
           self.fc = torch.nn.Linear(num_inputs, 1, bias=False)
           self.lif = snn.Leaky(
               beta=0.9,
               threshold=1.0,
               learn_beta=True,
               learn_threshold=True,
               spike_grad=surrogate.atan(),
               reset_mechanism="subtract"
           )

       def forward(self, x_spikes):
           # x_spikes: [Batch, TimeSteps, Inputs]
           mem = torch.zeros(x_spikes.size(0), 1)
           spk_out_list = []
           for t in range(x_spikes.size(1)):
               cur = self.fc(x_spikes[:, t])
               spk, mem = self.lif(cur, mem)
               spk_out_list.append(spk)
           return torch.stack(spk_out_list, dim=1)
   ```
3. **Pelatihan Warmup (Float32)**: Latih model selama 15 epoch pertama dengan presisi *floating-point* untuk menstabilkan bobot.
4. **Penerapan QAT (Format Q4.4)**: Latih model dengan simulasi kuantisasi *fixed-point* **Q4.4** (8-bit signed, 4 fractional bits) menggunakan *Straight-Through Estimator* (STE) agar tahan terhadap *quantization noise*.

### **Fase 2: Ekspor Bobot & Parameter ke Hardware Format**
1. Skalakan bobot terkuantisasi dari PyTorch ke format integer 8-bit Q4.4:
   $$\text{Weight}_{fixed} = \text{clamp}(\text{round}(\text{Weight}_{float} \times 16), -128, 127)$$
2. Simpan nilai bobot ke dalam file teks memori Hexadecimal/Binary (misalnya `weight_q44.mem`).

### **Fase 3: Perancangan RTL Hardware (Verilog)**
1. Buat file `lif_neuron.v` pada Vivado.
2. Implementasikan register $V_{mem}$, logika AND-gating sinaps, peluruhan *shift-and-subtract*, komparator $V_{th}$, dan *soft reset*.
3. Gunakan `$readmemh("weight_q44.mem", weight_mem)` untuk memuat bobot dari file ke memori internal.

### **Fase 4: Simulasi Testbench & Verifikasi Ekuivalensi (Vivado)**
1. Buat file *testbench* (`tb_lif_neuron.v`) di Vivado untuk menguji modul:
   * Deklarasikan sinyal `clk` (100 MHz) dan `rst_n`.
   * Berikan sekuens `spikes_in` biner hasil *rate encoding*.
2. Amati bentuk gelombang (*waveform*) pada **Vivado Behavioral Simulation**:
   * Verifikasi kenaikan $V_{mem}$ saat `spikes_in = 1`.
   * Verifikasi peluruhan eksponensial $V_{mem}$ saat `spikes_in = 0`.
   * Verifikasi kemunculan `spike_out = 1` saat $V_{mem} \ge V_{th}$ dan pemotongan nilai $V_{mem} = V_{mem} - V_{th}$.
3. **Hardware-Software Equivalence Check**: Bandingkan urutan `spike_out` dari simulasi Vivado dengan luaran tensor PyTorch (*golden model*). Hasil harus **100% ekuivalen**.

### **Fase 5: Sintesis, Implementasi & Analisis Resource di Vivado**
1. Jalankan **Run Synthesis** dan **Run Implementation** di Vivado untuk target FPGA (misal Xilinx Zynq-7000 / Artix-7).
2. Evaluasi laporan **Utilization Report** (pastikan konsumsi LUT, FF, dan DSP = 0).
3. Evaluasi laporan **Timing Report** (pastikan *Worst Negative Slack* / WNS $> 0$ ns pada frekuensi 100 MHz).

---

## 4. Struktur Direktori Proyek

```text
Test_LIF_Neuron/
├── Train/
│   ├── generate_dataset.py            # Pembangkitan data EKG & Direct Rate Spike Encoding
│   ├── train_qat.py                   # Pelatihan QAT PyTorch/snnTorch (Format Q4.4)
│   ├── export_weights.py              # Ekspor bobot ke file weight_q44.mem & vektor stimulus
│   ├── verify_hardware_equivalence.py # Verifikasi ekuivalensi bit-exact Python vs RTL
│   ├── best_lif_model.pt              # Model checkpoint PyTorch terlatih
│   ├── ecg_spikes_dataset.pt          # Dataset spike spatio-temporal [64, 30]
│   └── training_history.png           # Grafik kurva loss & akurasi
│
├── FPGA/
│   ├── weight_q44.mem                 # Parameter bobot 8-bit terkuantisasi
│   ├── input_spikes.mem               # Stimulus spike input 30-bit per slot waktu
│   ├── golden_spikes.mem              # Target luaran spike ekuivalen
│   ├── golden_vmem.mem                # Target luaran Vmem (16-bit Q12.4)
│   ├── lif_neuron.v                   # Modul RTL Verilog 1 Neuron LIF Digital (0 DSP)
│   ├── tb_lif_neuron.v                # File Testbench Simulasi Vivado
│   └── waveform_verification.png       # Grafik verifikasi bentuk gelombang Vmem & Spike
│
├── hasil_test.md                      # Laporan komprehensif hasil pengujian
└── README.md                          # Dokumentasi roadmap proyek
```

---

## 5. Lembar Ceklist Pengujian (*Verification Checklist*)

- [x] Data EKG berhasil di-encode ke matriks biner $[64 \times 30]$.
- [x] Model PyTorch `snnTorch` berhasil dilatih dengan QAT Q4.4 tanpa degradasi akurasi (100% test accuracy).
- [x] File `weight_q44.mem` berhasil diekspor.
- [x] Sintesis Verilog di Vivado bebas *warning* / *error*.
- [x] Simulasi Waveform menunjukkan integrasi, peluruhan, *fire*, dan *reset* yang benar.
- [x] Luaran *spike* simulasi Vivado match 100% dengan prediksi Python.
- [x] Desain digital murni dengan 0 DSP slice & timing siap untuk 100 MHz.
