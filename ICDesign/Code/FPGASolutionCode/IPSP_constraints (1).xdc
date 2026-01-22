################################################################################
## IPSP_constraints.xdc - Timing Constraints for IPSP Designs
## EEE3027 IC Design Labs
##
## Target: Artix-7 FPGA (xc7a35tcpg236-1)
## Clock: 50 MHz (20ns period) - achievable for both single-cycle and pipelined
################################################################################

# Primary clock constraint: 50 MHz
# Period = 20ns, 50% duty cycle (rise at 0ns, fall at 10ns)
create_clock -period 20.000 -name sys_clk_pin -waveform {0.000 10.000} [get_ports clk]

# Input jitter tolerance (accounts for clock source variations)
set_input_jitter sys_clk_pin 0.100

################################################################################
## For timing experiments, uncomment the following to test higher frequencies:
##
## 100 MHz (10ns period) - Pipelined should pass, Single-cycle may fail
## create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk]
##
## 125 MHz (8ns period) - Test maximum pipelined frequency
## create_clock -period 8.000 -name sys_clk_pin -waveform {0.000 4.000} [get_ports clk]
################################################################################
