# create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk]
create_clock -period 20.000 -name sys_clk_pin -waveform {0.000 10.000} [get_ports clk]

# Input clock should be clean - add jitter tolerance
set_input_jitter sys_clk_pin 0.100