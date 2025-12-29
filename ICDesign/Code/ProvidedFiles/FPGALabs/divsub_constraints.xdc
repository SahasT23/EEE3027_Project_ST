
create_clock -period 20.000 -name sys_clk -waveform {0.000 10.000} [get_ports clk]

set_input_jitter sys_clk 0.100
