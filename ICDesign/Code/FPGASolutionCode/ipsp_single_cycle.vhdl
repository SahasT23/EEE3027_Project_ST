library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fp_pkg.all;

entity ipsp_single_cycle is
  port (
    clk        : in  std_logic;
    rst        : in  std_logic;
    en         : in  std_logic;

    valid_in   : in  std_logic;
    a_in       : in  fp32;  -- coefficient
    x_in       : in  fp32;  -- input data
    y_in       : in  fp32;  -- previous partial sum

    valid_out  : out std_logic;
    a_out      : out fp32;  -- pass-through (aligned)
    x_out      : out fp32;  -- pass-through (aligned)
    y_out      : out fp32   -- new partial sum: a*x + y
  );
end entity;

architecture rtl of ipsp_single_cycle is
begin
  process(clk)
    -- Variables for combinational computation (evaluate immediately)
    variable product : fp32;
    variable sum     : fp32;
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- Synchronous reset: clear all outputs
        valid_out <= '0';
        a_out     <= (others => '0');
        x_out     <= (others => '0');
        y_out     <= (others => '0');
      elsif en = '1' then
        -- Compute MAC operation: y_out = a_in * x_in + y_in
        -- Step 1: Multiply coefficient by input data (Q16.16 * Q16.16)
        product := fp_mul(a_in, x_in);
        
        -- Step 2: Add previous partial sum with saturation
        sum := fp_add(product, y_in);
        
        -- Step 3: Register all outputs (appear on next clock cycle)
        valid_out <= valid_in;      -- Propagate valid signal
        a_out     <= a_in;          -- Pass-through for systolic chaining
        x_out     <= x_in;          -- Pass-through for systolic chaining
        y_out     <= sum;           -- Computed MAC result
      end if;
    end if;
  end process;
end architecture;