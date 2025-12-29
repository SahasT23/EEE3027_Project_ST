library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use std.textio.all;
use work.fp_pkg.all;

entity divsub_testbench_single_cycle is
end entity;

architecture tb of divsub_testbench_single_cycle is
  signal clk       : std_logic := '0';
  signal rst       : std_logic := '1';
  signal a_in,b_in,y_in : fp32 := (others => '0');
  signal valid_in  : std_logic := '0';
  signal x_out     : fp32;
  signal valid_out : std_logic;

  constant CLK_T : time := 20 ns;

  -- DUT
  component divsub is
    port (
      clk       : in  std_logic;
      rst       : in  std_logic;
      a_in      : in  fp32;
      b_in      : in  fp32;
      y_in      : in  fp32;
      valid_in  : in  std_logic;
      x_out     : out fp32;
      valid_out : out std_logic
    );
  end component;
begin
  clk <= not clk after CLK_T/2;

  dut: divsub
    port map (
      clk, rst, a_in, b_in, y_in, valid_in, x_out, valid_out
    );

  stim: process
    variable exp_r : real;
    variable got_r : real;
  begin
    -- Reset
    rst <= '1';
    valid_in <= '0';
    wait for 10*CLK_T;
    rst <= '0';
    wait for CLK_T;
    wait until rising_edge(clk) and rst='0';


    -- Test 1: (3.0 - 1.0) / 2.0 = 1.0
    a_in <= real_to_fp(2.0);
    b_in <= real_to_fp(3.0);
    y_in <= real_to_fp(1.0);
    valid_in <= '1';
    wait for CLK_T;
    valid_in <= '0';

    wait until rising_edge(clk) and valid_out='1';
    exp_r := (3.0 - 1.0)/2.0;
    got_r := fp_to_real(x_out);
      report "Test1 mismatch: got=" & real'image(got_r) & " exp=" & real'image(exp_r)
      severity error;

    -- Test 2: divide by zero -> 0.0
    a_in <= real_to_fp(0.0);
    b_in <= real_to_fp(5.0);
    y_in <= real_to_fp(0.0);
    valid_in <= '1';
    wait for CLK_T;
    valid_in <= '0';

    wait until rising_edge(clk) and valid_out='1';
    got_r := fp_to_real(x_out);
      report "Test2 mismatch (div0 policy): got=" & real'image(got_r)
      severity error;

    -- Test 3: negative values: (1.0 - 3.0)/2.0 = -1.0
    a_in <= real_to_fp(2.0);
    b_in <= real_to_fp(1.0);
    y_in <= real_to_fp(3.0);
    valid_in <= '1';
    wait for CLK_T;
    valid_in <= '0';

    wait until rising_edge(clk) and valid_out='1';
    exp_r := (1.0 - 3.0)/2.0;
    got_r := fp_to_real(x_out);
      report "Test3 mismatch: got=" & real'image(got_r) & " exp=" & real'image(exp_r)
      severity error;

    report "divsub_testbench_single_cycle: ALL TESTS PASSED" severity note;
    wait;
  end process;
end architecture;
