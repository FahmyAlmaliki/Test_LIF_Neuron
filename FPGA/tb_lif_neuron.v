`timescale 1ns / 1ps
// =============================================================================
// File    : tb_lif_neuron.v
// Module  : tb_lif_neuron
// Author  : Digital SNN Hardware Design Team
// Project : Testbench Verifikasi 1 Neuron LIF Digital (Vivado Behavioral Sim)
// =============================================================================
// Deskripsi:
// Menguji modul 'lif_neuron' pada frekuensi 100 MHz (Clock Period = 10 ns):
// 1. Memuat file input_spikes.mem (Stimulus 64 slot waktu)
// 2. Memuat golden_spikes.mem & golden_vmem.mem hasil kalkulasi software Python
// 3. Membandingkan luaran hardware (spike_out & vmem_out) pada setiap siklus
// 4. Memastikan 100% Hardware-Software Equivalence Check (Sesuai Roadmap Fase 4)
// =============================================================================

module tb_lif_neuron;

    // Parameter rangkaian
    parameter integer NUM_INPUTS   = 30;
    parameter integer WEIGHT_WIDTH = 8;
    parameter integer VMEM_WIDTH   = 16;
    parameter integer DECAY_SHIFT  = 3;
    parameter signed [VMEM_WIDTH-1:0] V_TH = 16'sd16;
    parameter integer NUM_SLOTS    = 64;
    parameter integer CLK_PERIOD   = 10; // 100 MHz

    // Sinyal stimulus testbench
    reg                      clk;
    reg                      rst_n;
    reg                      enable;
    reg [NUM_INPUTS-1:0]     spike_in;

    // Sinyal luaran Unit Under Test (UUT)
    wire                     spike_out;
    wire signed [VMEM_WIDTH-1:0] vmem_out;

    // Memori vektor uji
    reg [NUM_INPUTS-1:0]         mem_spikes_in  [0:NUM_SLOTS-1];
    reg                          mem_golden_spk [0:NUM_SLOTS-1];
    reg signed [VMEM_WIDTH-1:0]  mem_golden_vm  [0:NUM_SLOTS-1];

    // Variabel loop dan verifikasi
    integer slot;
    integer spike_match_count;
    integer vmem_match_count;
    integer total_spikes_uut;
    integer total_spikes_golden;

    // =========================================================================
    // Instansiasi Modul lif_neuron (Unit Under Test / UUT)
    // =========================================================================
    lif_neuron #(
        .NUM_INPUTS(NUM_INPUTS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH),
        .DECAY_SHIFT(DECAY_SHIFT),
        .V_TH(V_TH),
        .WEIGHT_FILE("weight_q44.mem")
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .spike_in(spike_in),
        .spike_out(spike_out),
        .vmem_out(vmem_out)
    );

    // =========================================================================
    // Generator Sinyal Detak 100 MHz (Periode = 10 ns)
    // =========================================================================
    always #(CLK_PERIOD / 2) clk = ~clk;

    // =========================================================================
    // Alur Simulasi Utama
    // =========================================================================
    initial begin
        // Perekaman gelombang VCD (opsional, untuk GTKWave / ModelSim / Vivado)
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_lif_neuron);

        // Inisialisasi awal
        clk                 = 1'b0;
        rst_n               = 1'b0;
        enable              = 1'b0;
        spike_in            = {NUM_INPUTS{1'b0}};
        spike_match_count   = 0;
        vmem_match_count    = 0;
        total_spikes_uut    = 0;
        total_spikes_golden = 0;

        $display("=================================================================");
        $display("   TESTBENCH VERIFIKASI 1 NEURON DIGITAL LIF (VIVADO SIMULASI)   ");
        $display("=================================================================");

        // Memuat vektor stimulus dan ekspektasi
        $readmemh("input_spikes.mem", mem_spikes_in);
        $readmemh("golden_spikes.mem", mem_golden_spk);
        $readmemh("golden_vmem.mem",   mem_golden_vm);

        // Reset rangkaian selama 2 siklus clock
        #(CLK_PERIOD * 2);
        rst_n  = 1'b1;
        enable = 1'b1;
        #(CLK_PERIOD);

        $display("[*] Reset selesai. Memulai injeksi %0d slot stimulus...", NUM_SLOTS);
        $display("-----------------------------------------------------------------");
        $display("Slot | Spike_In (Hex) | Spike UUT | Golden Spk | Vmem UUT | Golden Vm | Status");
        $display("-----------------------------------------------------------------");

        // Loop evaluasi 64 slot waktu
        for (slot = 0; slot < NUM_SLOTS; slot = slot + 1) begin
            // 1. Berikan spike input pada slot saat ini
            spike_in = mem_spikes_in[slot];

            // 2. Evaluasi Spike_Out sesaat sebelum clock naik (sampling window)
            #(CLK_PERIOD - 2);

            if (spike_out === mem_golden_spk[slot]) begin
                spike_match_count = spike_match_count + 1;
            end else begin
                $display("[ERROR SPIKE MISMATCH] Slot %02d: UUT=%b != Golden=%b",
                         slot, spike_out, mem_golden_spk[slot]);
            end

            if (spike_out == 1'b1) total_spikes_uut = total_spikes_uut + 1;
            if (mem_golden_spk[slot] == 1'b1) total_spikes_golden = total_spikes_golden + 1;

            // 3. Tepi naik clock terjadi: Vmem diperbarui
            #2; 
            // Tunggu sedikit setelah clock edge untuk propagasi register
            #1;

            // 4. Verifikasi nilai Vmem terhadap model golden Python
            if (vmem_out === mem_golden_vm[slot]) begin
                vmem_match_count = vmem_match_count + 1;
                $display(" %02d  |   30'h%08X  |     %b     |     %b      |  %5d   |  %5d    |  PASS",
                         slot, spike_in, spike_out, mem_golden_spk[slot], vmem_out, mem_golden_vm[slot]);
            end else begin
                $display(" %02d  |   30'h%08X  |     %b     |     %b      |  %5d   |  %5d    |  FAIL",
                         slot, spike_in, spike_out, mem_golden_spk[slot], vmem_out, mem_golden_vm[slot]);
            end

            // Selesaikan sisa siklus detak
            #(CLK_PERIOD - 1);
        end

        // Nonaktifkan rangkaian
        enable   = 1'b0;
        spike_in = {NUM_INPUTS{1'b0}};
        #(CLK_PERIOD * 2);

        // =====================================================================
        // Ringkasan Hasil Verifikasi
        // =====================================================================
        $display("=================================================================");
        $display("                    LAPORAN VERIFIKASI AKHIR                     ");
        $display("=================================================================");
        $display("Total Siklus Uji         : %0d", NUM_SLOTS);
        $display("Total Spike UUT          : %0d", total_spikes_uut);
        $display("Total Spike Golden Model : %0d", total_spikes_golden);
        $display("Spike Match Rate         : %0d / %0d (%.1f%%)",
                 spike_match_count, NUM_SLOTS, (spike_match_count * 100.0) / NUM_SLOTS);
        $display("Vmem Match Rate          : %0d / %0d (%.1f%%)",
                 vmem_match_count, NUM_SLOTS, (vmem_match_count * 100.0) / NUM_SLOTS);

        if ((spike_match_count == NUM_SLOTS) && (vmem_match_count == NUM_SLOTS)) begin
            $display("\n>>> STATUS: [PASS] 100% BIT-EXACT HARDWARE-SOFTWARE EQUIVALENCE! <<<");
            $display(">>> Desain siap untuk sintesis dan implementasi FPGA di Vivado. <<<\n");
        end else begin
            $display("\n>>> STATUS: [FAIL] Ditemukan perbedaan antara RTL dan Golden Model! <<<\n");
        end
        $display("=================================================================");

        $finish;
    end

endmodule

