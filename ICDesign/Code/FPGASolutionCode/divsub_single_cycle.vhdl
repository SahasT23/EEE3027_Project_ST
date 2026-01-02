library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

use work.fp_pkg.all;

entity divsub is
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;               
    a_in      : in  fp32;                 -- Q16.16 divisor
    b_in      : in  fp32;                 -- Q16.16 minuend
    y_in      : in  fp32;                 -- Q16.16 subtrahend
    valid_in  : in  std_logic;              
    x_out     : out fp32;                 -- Q16.16 result: (b - y) / a
    valid_out : out std_logic                
  );
end entity;

architecture rtl of divsub is

  -- Fixed-point division function for Q16.16 format
  -- Computes: result = numerator / denominator in Q16.16
  function fp_divide(a : fp32; b : fp32) return fp32 is
    variable num_s  : signed(47 downto 0);  -- 48-bit for scaled numerator
    variable den_s  : signed(31 downto 0);  -- 32-bit denominator
    variable quo_s  : signed(47 downto 0);  -- 48-bit quotient before resize
  begin
    den_s := signed(b);
    
    -- Handle divide-by-zero: return 0
    if den_s = 0 then
      return (others => '0');
    else
      -- Scale numerator left by 16 bits to maintain Q16.16 precision
      -- (a / 2^16) / (b / 2^16) = a/b, but we need (a/b) * 2^16 for Q16.16
      -- So: ((a << 16) / b) gives us the correctly scaled result
      num_s := resize(signed(a), 48) sll 16;
      quo_s := num_s / resize(den_s, 48);
      return resize(quo_s, 32);
    end if;
  end function;

  -- Output registers
  signal x_r : fp32 := (others => '0');
  signal v_r : std_logic := '0';

  -- Intermediate subtraction result (combinational)
  signal diff_q : fp32;

begin

  -- Step 1: Compute subtraction (b - y) combinationally
  diff_q <= fp_sub(b_in, y_in);

  -- Step 2: Register the division result for 1-cycle latency
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        x_r <= (others => '0');
        v_r <= '0';
      else
        -- Compute x = (b - y) / a using the fp_divide function
        x_r <= fp_divide(diff_q, a_in);
        v_r <= valid_in;
      end if;
    end if;
  end process;

  -- Output assignments
  x_out     <= x_r;
  valid_out <= v_r;

end architecture;