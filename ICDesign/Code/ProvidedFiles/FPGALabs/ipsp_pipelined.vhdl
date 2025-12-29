library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fp_pkg.all;

entity ipsp_pipelined is
  port (
    clk        : in  std_logic;
    rst        : in  std_logic;
    en         : in  std_logic;

    valid_in   : in  std_logic;
    a_in       : in  fp32;
    x_in       : in  fp32;
    y_in       : in  fp32;

    valid_out  : out std_logic;
    a_out      : out fp32;
    x_out      : out fp32;
    y_out      : out fp32
  );
end entity;

architecture rtl of ipsp_pipelined is
  -- Stage registers (declare and comment each stage)
  -- S1: input regs
  signal s1_valid : std_logic;
  signal s1_a, s1_x, s1_y : fp32;

  -- S2: multiply
  signal s2_valid : std_logic;
  signal s2_a, s2_x : fp32;
  signal s2_prod    : fp32;

  -- S3: register product
  signal s3_valid : std_logic;
  signal s3_a, s3_x : fp32;
  signal s3_prod, s3_y : fp32;

  -- S4: add and output
  signal s4_valid : std_logic;
  signal s4_a, s4_x, s4_y : fp32;
begin

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- TODO: reset all stage registers
        null;
      elsif en = '1' then
        -- === Stage 1: register inputs ===
        -- TODO

        -- === Stage 2: multiply ===
        -- TODO

        -- === Stage 3: register product (+ align y) ===
        -- TODO

        -- === Stage 4: add and drive outputs ===
        -- TODO
      end if;
    end if;
  end process;

  -- Outputs
  valid_out <= s4_valid;
  a_out     <= s4_a;
  x_out     <= s4_x;
  y_out     <= s4_y;

end architecture;