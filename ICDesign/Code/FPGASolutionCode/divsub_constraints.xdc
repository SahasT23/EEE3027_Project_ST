################################################################################
## divsub_constraints.xdc - Timing Constraints for Multi-Cycle DivSub
## EEE3027 IC Design Labs - Lab 5 Version 2
##
## Target: Artix-7 FPGA (xc7a35tcpg236-1)
## Clock: 50 MHz (20ns period)
##
## Multi-cycle DivSub SHOULD pass timing at 50 MHz because:
## - Division is spread across 48 clock cycles
## - Each cycle only performs: shift + compare + conditional subtract
## - Critical path is much shorter than single-cycle division
################################################################################

# Primary clock: 50 MHz
create_clock -period 20.000 -name sys_clk_pin -waveform {0.000 10.000} [get_ports clk]

# Input jitter
set_input_jitter sys_clk_pin 0.100
