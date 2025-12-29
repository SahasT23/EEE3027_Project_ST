library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

use work.fp_pkg.all;

entity divsub is
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;               
    a_in      : in  fp32;                 -- Q16.16
    b_in      : in  fp32;                 -- Q16.16
    y_in      : in  fp32;                 -- Q16.16
    valid_in  : in  std_logic;              
    x_out     : out fp32;                 -- Q16.16 result
    valid_out : out std_logic                
  );
end entity;

architecture rtl of divsub is

  function fp_divide(a : fp32; b : fp32) return fp32 is
    variable num_s  : signed(--what goes here downto 0);
    variable den_s  : signed(--what goes here  downto 0);
    variable quo_s  : signed(--what goes here  downto 0);
  begin
    den_s := signed(b);
    if den_s = 0 then
      return (others => '0');  -- Policy: return 0 on divide-by-zero (explicit constant from package)
    else
      -- Scale numerator by 2^16 to keep Q16.16 after integer division
      num_s := resize(signed(a), 48) sll 16;
      quo_s := resize(num_s / den_s, 32);
      return quo_s;
    end if;
  end function;

  signal x_r        : fp32 := (others => '0');
  signal v_r        : std_logic := '0';

  signal diff_q     : fp32;

begin

  diff_q <= fp_sub(b_in, y_in);

  -- Register outputs for a clean 1-cycle latency
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        x_r <= (others => '0');
        v_r <= '0';
      else
        -- Division via local helper (keeps everything in fp_data)
        x_r <= --what goes here?---;
        v_r <= valid_in;
      end if;
    end if;
  end process;

  x_out     <= x_r;
  valid_out <= v_r;

end architecture;
