################################################################################
## array_constraints.xdc - Timing Constraints for Systolic Array
## EEE3027 IC Design Labs - Lab 6
##
## Target: Artix-7 FPGA (xc7a35tcpg236-1)
## Clock: 50 MHz (20ns period)
##
## The full systolic array should meet timing at 50 MHz because:
## - DivSub uses multi-cycle division (48 cycles)
## - IPSP uses 4-stage pipelining
## - Controller FSM has simple sequential logic
################################################################################

# Primary clock: 50 MHz (20ns period)
create_clock -period 20.000 -name sys_clk_pin -waveform {0.000 10.000} [get_ports clk]

# Input jitter tolerance
set_input_jitter sys_clk_pin 0.100

################################################################################
## Optional: For higher frequency experiments
##
## 100 MHz (10ns period):
## create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk]
################################################################################
