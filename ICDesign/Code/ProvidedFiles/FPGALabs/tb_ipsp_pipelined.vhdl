library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fp_pkg.all;

entity tb_ipsp_pipelined is
end;

architecture sim of tb_ipsp_pipelined is
  signal clk, rst, en, valid_in, valid_out : std_logic := '0';
  signal a_in, x_in, y_in : fp32 := (others => '0');
  signal a_out, x_out, y_out : fp32;
begin
  dut: entity work.ipsp_pipelined
    port map (
      clk=>clk, rst=>rst, en=>en,
      valid_in=>valid_in, a_in=>a_in, x_in=>x_in, y_in=>y_in,
      valid_out=>valid_out, a_out=>a_out, x_out=>x_out, y_out=>y_out
    );

  -- 100 MHz
  clk <= not clk after 5 ns;

  process
  begin
    rst <= '1'; en <= '0'; wait for 20 ns;
    rst <= '0'; en <= '1'; wait for 10 ns;

    -- Stimulus 1: a=1.0, x=0.5, y=0.25 => y=0.75
    a_in <= FP_ONE;
    x_in <= real_to_fp(0.5);   -- 0.5
    y_in <= real_to_fp(0.25);   -- 0.25
    valid_in <= '1'; wait for 10 ns; valid_in <= '0';

    -- Stimulus 2: a=2.0, x=0.25, y=0.25 => y=0.75
    wait for 20 ns;
    a_in <= real_to_fp(2.0);  -- 2.0
    x_in <= real_to_fp(0.25);      -- 0.25
    y_in <= real_to_fp(0.25);      -- 0.25
    valid_in <= '1'; wait for 10 ns; valid_in <= '0';

    -- Allow pipeline to drain
    wait for 100 ns;

    report "Last y_out (real) = " & real'image(fp_to_real(y_out));
    report "SIM DONE";
    wait;
  end process;
end;