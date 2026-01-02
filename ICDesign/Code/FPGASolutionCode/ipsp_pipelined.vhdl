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
  -- ===== Stage 1 Registers: Input capture =====
  signal s1_valid : std_logic;
  signal s1_a, s1_x, s1_y : fp32;

  -- ===== Stage 2 Registers: After multiplication =====
  signal s2_valid : std_logic;
  signal s2_a, s2_x, s2_y : fp32;  -- y propagates for alignment
  signal s2_prod : fp32;            -- multiplication result

  -- ===== Stage 3 Registers: Product registration =====
  signal s3_valid : std_logic;
  signal s3_a, s3_x, s3_y : fp32;
  signal s3_prod : fp32;

  -- ===== Stage 4 Registers: After addition (outputs) =====
  signal s4_valid : std_logic;
  signal s4_a, s4_x, s4_y : fp32;

begin

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- ===== Reset all pipeline registers =====
        -- Stage 1
        s1_valid <= '0';
        s1_a     <= (others => '0');
        s1_x     <= (others => '0');
        s1_y     <= (others => '0');
        
        -- Stage 2
        s2_valid <= '0';
        s2_a     <= (others => '0');
        s2_x     <= (others => '0');
        s2_y     <= (others => '0');
        s2_prod  <= (others => '0');
        
        -- Stage 3
        s3_valid <= '0';
        s3_a     <= (others => '0');
        s3_x     <= (others => '0');
        s3_y     <= (others => '0');
        s3_prod  <= (others => '0');
        
        -- Stage 4
        s4_valid <= '0';
        s4_a     <= (others => '0');
        s4_x     <= (others => '0');
        s4_y     <= (others => '0');

      elsif en = '1' then
        -- ===== Stage 1: Register all inputs =====
        -- Purpose: Capture inputs, create clean timing boundary
        s1_valid <= valid_in;
        s1_a     <= a_in;
        s1_x     <= x_in;
        s1_y     <= y_in;

        -- ===== Stage 2: Perform multiplication =====
        -- Purpose: Compute a*x, propagate y for later alignment
        s2_valid <= s1_valid;
        s2_a     <= s1_a;
        s2_x     <= s1_x;
        s2_y     <= s1_y;                    -- y delayed to align with product
        s2_prod  <= fp_mul(s1_a, s1_x);      -- MAC operation part 1

        -- ===== Stage 3: Register multiplication result =====
        -- Purpose: Break timing path after DSP48E1 multiplier
        s3_valid <= s2_valid;
        s3_a     <= s2_a;
        s3_x     <= s2_x;
        s3_y     <= s2_y;
        s3_prod  <= s2_prod;

        -- ===== Stage 4: Perform addition and output =====
        -- Purpose: Compute product + y, register final outputs
        s4_valid <= s3_valid;
        s4_a     <= s3_a;
        s4_x     <= s3_x;
        s4_y     <= fp_add(s3_prod, s3_y);   -- MAC operation part 2

      end if;
    end if;
  end process;

  -- ===== Output assignments =====
  valid_out <= s4_valid;
  a_out     <= s4_a;
  x_out     <= s4_x;
  y_out     <= s4_y;

end architecture;