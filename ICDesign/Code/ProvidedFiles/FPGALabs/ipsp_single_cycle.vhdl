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
    y_out      : out fp32   -- new partial sum
  );
end entity;

architecture rtl of ipsp_single_cycle is
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        valid_out <= '0';
        a_out     <= (others => '0');
        x_out     <= (others => '0');
        y_out     <= (others => '0');
      elsif en = '1' then
        -- TODO:
        -- 1) Compute product = fp_mul(a_in, x_in)
        -- 2) Compute sum     = fp_add(product, y_in)
        -- 3) Register outputs so that when valid_in='1' this cycle,
        --    valid_out='1' next cycle with corresponding y_out
        -- 4) Pass-through a_out and x_out aligned with y_out
      end if;
    end if;
  end process;
end architecture;