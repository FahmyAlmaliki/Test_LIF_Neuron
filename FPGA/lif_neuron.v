`timescale 1ns / 1ps
// =============================================================================
// File    : lif_neuron.v
// Module  : lif_neuron
// Author  : Digital SNN Hardware Design Team
// Project : Single Digital Leaky Integrate-and-Fire (LIF) Neuron for FPGA
// Ref     : Sensors 2021 (Asghar et al.) & SNN4FPGA Framework
// =============================================================================
// Deskripsi:
// Arsitektur Hardware Digital Murni (All-Digital) untuk 1 Neuron LIF:
// 1. Sinaps / Input Selektor: Gerbang AND-gating (0 DSP Slice).
// 2. Akumulator / Adder Tree: Menjumlahkan seluruh sinaps aktif.
// 3. Unit Peluruhan (Leakage Unit): Operasi Shift-and-Subtract:
//       V_decay[n] = V_mem[n] - (V_mem[n] >>> DECAY_SHIFT)
// 4. Komparator Ambang Batas: Membandingkan V_sum >= V_TH.
// 5. Soft Reset Mechanism:
//       V_mem[n+1] = Spike_Out[n] ? (V_sum[n] - V_TH) : V_sum[n]
// =============================================================================

module lif_neuron #(
    parameter integer NUM_INPUTS   = 30,                 // Jumlah kanal sinaps masukan (C = 30)
    parameter integer WEIGHT_WIDTH = 8,                  // Lebar bit bobot (Q4.4 signed: 8-bit)
    parameter integer VMEM_WIDTH   = 16,                 // Lebar bit potensial membran (Q12.4 signed: 16-bit)
    parameter integer DECAY_SHIFT  = 3,                  // Faktor peluruhan beta = 1 - 2^-3 = 0.875
    parameter signed [VMEM_WIDTH-1:0] V_TH = 16'sd16,    // Ambang batas tembak (1.0 * 16 = 16)
    parameter WEIGHT_FILE          = "weight_q44.mem"    // File heksadesimal bobot terlatih
)(
    input  wire                         clk,             // Sinyal detak sinkron (100 MHz)
    input  wire                         rst_n,           // Asynchronous active-low reset
    input  wire                         enable,          // Enable sinyal untuk komputasi time-step
    input  wire [NUM_INPUTS-1:0]        spike_in,        // Masukan spike biner spatio-temporal [29:0]
    output wire                         spike_out,       // Luaran spike biner (1 = firing, 0 = silent)
    output wire signed [VMEM_WIDTH-1:0] vmem_out         // Nilai potensial membran aktif (Vmem)
);

    // =========================================================================
    // 1. Memori Bobot Sinaps (Internal Weight Memory)
    // =========================================================================
    // Array penyimpanan bobot terkuantisasi Q4.4 (8-bit signed)
    reg signed [WEIGHT_WIDTH-1:0] weight_mem [0:NUM_INPUTS-1];

    initial begin
        // Memuat nilai bobot hasil pelatihan PyTorch QAT dari file heksadesimal
        $readmemh(WEIGHT_FILE, weight_mem);
    end

    // =========================================================================
    // 2. Sinaps Selektor & Pohon Penjumlah (AND-Gating & Adder Tree)
    // =========================================================================
    // Jika Spike_In[i] == 1, bobot W[i] diteruskan (Sign-Extended ke VMEM_WIDTH).
    // Jika Spike_In[i] == 0, nilai 0 diteruskan.
    // Menghilangkan penggunaan multiplier/DSP slice (0 DSP blocks).
    integer i;
    reg signed [VMEM_WIDTH-1:0] acc_val;

    always @(*) begin
        acc_val = {VMEM_WIDTH{1'b0}};
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            if (spike_in[i]) begin
                // Sign-extension bobot 8-bit ke 16-bit
                acc_val = acc_val + $signed({{ (VMEM_WIDTH - WEIGHT_WIDTH){weight_mem[i][WEIGHT_WIDTH-1]} }, weight_mem[i]});
            end
        end
    end

    // =========================================================================
    // 3. Register Potensial Membran (Membrane Register)
    // =========================================================================
    reg signed [VMEM_WIDTH-1:0] vmem_reg;

    // =========================================================================
    // 4. Unit Peluruhan (Leakage Unit: Shift-and-Subtract)
    // =========================================================================
    // Operasi aritmatika pergeseran ke kanan (>>> menjaga tanda bilangan negatif)
    // V_decay = V_mem - (V_mem >>> DECAY_SHIFT)
    wire signed [VMEM_WIDTH-1:0] v_decay;
    assign v_decay = vmem_reg - (vmem_reg >>> DECAY_SHIFT);

    // =========================================================================
    // 5. Integrasi Arus & Ambang Batas (Summing & Threshold Comparator)
    // =========================================================================
    // V_sum[n] = V_decay[n] + Accumulator
    wire signed [VMEM_WIDTH-1:0] v_sum;
    assign v_sum = enable ? (v_decay + acc_val) : v_decay;

    // Komparator: Spike_Out = 1 jika V_sum >= V_TH
    wire spike_condition;
    assign spike_condition = (v_sum >= V_TH);
    assign spike_out       = enable ? spike_condition : 1'b0;

    // =========================================================================
    // 6. Mekanisme Soft Reset & Pembaruan Potensial Membran
    // =========================================================================
    // Soft Reset: Saat neuron menembak (Spike_Out = 1), V_mem dikurangi sebesar V_TH
    // V_next = Spike_Out ? (V_sum - V_TH) : V_sum
    wire signed [VMEM_WIDTH-1:0] v_next;
    assign v_next = spike_condition ? (v_sum - V_TH) : v_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vmem_reg <= {VMEM_WIDTH{1'b0}};
        end else if (enable) begin
            vmem_reg <= v_next;
        end
    end

    // Keluaran port vmem_out mencerminkan potensial membran yang tersimpan
    assign vmem_out = vmem_reg;

endmodule

