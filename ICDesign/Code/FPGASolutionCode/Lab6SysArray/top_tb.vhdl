--------------------------------------------------------------------------------
-- top_tb.vhdl - Testbench for Systolic Band Solver
-- EEE3027 IC Design Labs - Lab 6
--
-- Tests the complete systolic array with 3 example cases
-- Each case: x = (b - y_feedback) / a_divsub
--            y_result = a_ipsp * x + y_ipsp_in
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fp_pkg.all;

entity systolic_band_solver_tb is
end;

architecture sim of systolic_band_solver_tb is
  -- Clock and control
  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal en    : std_logic := '0';
  signal start : std_logic := '0';

  -- Inputs to controller
  signal a_divsub   : fp32 := (others => '0');
  signal b_in       : fp32 := (others => '0');
  signal y_feedback : fp32 := (others => '0');
  signal a_ipsp     : fp32 := (others => '0');
  signal y_ipsp_in  : fp32 := (others => '0');

  -- Outputs from controller
  signal done     : std_logic;
  signal x_result : fp32;
  signal y_result : fp32;

  -- Clock period
  constant CLK_PERIOD : time := 10 ns;  -- 100 MHz

begin

  ----------------------------------------------------------------------------
  -- Instantiate Device Under Test
  ----------------------------------------------------------------------------
  dut: entity work.systolic_band_solver
    port map (
      clk        => clk,
      rst        => rst,
      en         => en,
      start      => start,
      a_divsub   => a_divsub,
      b_in       => b_in,
      y_feedback => y_feedback,
      a_ipsp     => a_ipsp,
      y_ipsp_in  => y_ipsp_in,
      done       => done,
      x_result   => x_result,
      y_result   => y_result
    );

  -- 100 MHz clock
  clk <= not clk after CLK_PERIOD / 2;

  ----------------------------------------------------------------------------
  -- Stimulus Process
  ----------------------------------------------------------------------------
  process
  begin
    report "=== SYSTOLIC ARRAY TESTBENCH START ===" severity note;
    
    -- Extended reset for post-implementation simulation
    rst <= '1';
    en  <= '0';
    wait for 200 ns;  -- Longer reset for post-implementation
    
    rst <= '0';
    en  <= '1';
    wait for 50 ns;

    --------------------------------------------------------------------------
    -- Test Case 1:
    -- DivSub: x = (b - y_fb) / a = (10 - 4) / 2 = 3.0
    -- IPSP:   y = a*x + y_in = 1.0 * 3.0 + 0.0 = 3.0
    --------------------------------------------------------------------------
    report "=== Test 1: DivSub(10-4)/2=3, IPSP(1*3+0)=3 ===" severity note;
    
    a_divsub   <= real_to_fp(2.0);   -- Divisor
    b_in       <= real_to_fp(10.0);  -- b value
    y_feedback <= real_to_fp(4.0);   -- y to subtract
    a_ipsp     <= FP_ONE;            -- IPSP coefficient
    y_ipsp_in  <= FP_ZERO;           -- IPSP y input
    
    start <= '1';
    wait for CLK_PERIOD;
    start <= '0';

    -- Wait for done signal
    wait until done = '1' for 2 ms;
    
    if done = '1' then
      report "Test 1 x_result = " & real'image(fp_to_real(x_result)) & " (expected 3.0)";
      report "Test 1 y_result = " & real'image(fp_to_real(y_result)) & " (expected 3.0)";
    else
      report "Test 1 TIMEOUT!" severity error;
    end if;

    wait for 100 ns;

    --------------------------------------------------------------------------
    -- Test Case 2:
    -- DivSub: x = (8 - 2) / 3 = 2.0
    -- IPSP:   y = 2.0 * 2.0 + 1.0 = 5.0
    --------------------------------------------------------------------------
    report "=== Test 2: DivSub(8-2)/3=2, IPSP(2*2+1)=5 ===" severity note;
    
    a_divsub   <= real_to_fp(3.0);
    b_in       <= real_to_fp(8.0);
    y_feedback <= real_to_fp(2.0);
    a_ipsp     <= real_to_fp(2.0);
    y_ipsp_in  <= FP_ONE;
    
    start <= '1';
    wait for CLK_PERIOD;
    start <= '0';

    wait until done = '1' for 2 ms;
    
    if done = '1' then
      report "Test 2 x_result = " & real'image(fp_to_real(x_result)) & " (expected 2.0)";
      report "Test 2 y_result = " & real'image(fp_to_real(y_result)) & " (expected 5.0)";
    else
      report "Test 2 TIMEOUT!" severity error;
    end if;

    wait for 100 ns;

    --------------------------------------------------------------------------
    -- Test Case 3:
    -- DivSub: x = (12 - 0) / 4 = 3.0
    -- IPSP:   y = 0.5 * 3.0 + 2.5 = 4.0
    --------------------------------------------------------------------------
    report "=== Test 3: DivSub(12-0)/4=3, IPSP(0.5*3+2.5)=4 ===" severity note;
    
    a_divsub   <= real_to_fp(4.0);
    b_in       <= real_to_fp(12.0);
    y_feedback <= FP_ZERO;
    a_ipsp     <= real_to_fp(0.5);
    y_ipsp_in  <= real_to_fp(2.5);
    
    start <= '1';
    wait for CLK_PERIOD;
    start <= '0';

    wait until done = '1' for 2 ms;
    
    if done = '1' then
      report "Test 3 x_result = " & real'image(fp_to_real(x_result)) & " (expected 3.0)";
      report "Test 3 y_result = " & real'image(fp_to_real(y_result)) & " (expected 4.0)";
    else
      report "Test 3 TIMEOUT!" severity error;
    end if;

    wait for 200 ns;
    
    report "=== SYSTOLIC ARRAY TESTBENCH COMPLETE ===" severity note;
    wait;
  end process;

end;
