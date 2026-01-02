library IEEE;                                                -- Standard libraries
use IEEE.STD_LOGIC_1164.ALL;                                 -- std_logic, std_logic_vector
use IEEE.NUMERIC_STD.ALL;                                    
use work.fp_pkg.all;                                
                                                            

entity divsub is
  port (
    clk       : in  std_logic;                              -- Clock
    rst       : in  std_logic;                              -- Synchronous reset (active '1')
    enable    : in  std_logic;                              -- Enable FIFO write
    a_in      : in  fp32;                                   -- Q16.16 divisor (two's complement)
    b_in      : in  fp32;                                   -- Q16.16 minuend
    y_in      : in  fp32;                                   -- Q16.16 subtrahend
    x_out     : out fp32;                                   -- Q16.16 result: (b_in - y_in) / a_in
    valid_in  : in  std_logic;                              
    valid_out : out std_logic                               
  );
end entity divsub;

architecture simple of divsub is

  ----------------------------------------------------------------------------
  -- Simple FIFO to decouple input from divider
  -- Allows continuous input while divider is busy
  ----------------------------------------------------------------------------
  type fifo_data_t is record
    a : fp32;
    b : fp32;
    y : fp32;
  end record;

  constant FIFO_DEPTH : integer := 32;                       -- Small queue
  type fifo_array_t is array (0 to FIFO_DEPTH-1) of fifo_data_t;
  signal fifo   : fifo_array_t;
  signal wr_ptr : integer range 0 to FIFO_DEPTH-1 := 0;      -- Write pointer
  signal rd_ptr : integer range 0 to FIFO_DEPTH-1 := 0;      -- Read pointer
  signal count  : integer range 0 to FIFO_DEPTH := 0;        -- Number of entries

  ----------------------------------------------------------------------------
  -- Control FSM - 4 states for division operation
  ----------------------------------------------------------------------------
  type state_t is (IDLE, PROCESS_DATA, DIVIDING, OUTPUT_RESULT);
  signal state : state_t := IDLE;

  ----------------------------------------------------------------------------
  -- Latched work registers - hold current operation's inputs
  ----------------------------------------------------------------------------
  signal work_a, work_b, work_y : fp32 := FP_ZERO;          
  signal div_sign : std_logic := '0';                        -- Sign of result: sign(b-y) XOR sign(a)

  ----------------------------------------------------------------------------
  -- Divider internals (unsigned magnitude division; sign applied at the end)
  -- We build a 48-bit dividend = |b-y| << 16 to preserve Q16.16 fractional bits
  ----------------------------------------------------------------------------
  signal dividend_reg  : unsigned(47 downto 0) := (others => '0'); -- 48-bit dividend (scaled)
  signal divisor_reg   : unsigned(31 downto 0) := (others => '0'); -- 32-bit divisor magnitude
  signal divisor_ext   : unsigned(32 downto 0) := (others => '0'); -- 33-bit for compare/subtract
  signal quotient_reg  : unsigned(31 downto 0) := (others => '0'); -- 32-bit Q16.16 quotient magnitude
  signal remainder_reg : unsigned(32 downto 0) := (others => '0'); -- 33-bit remainder (guard bit)
  signal div_counter   : integer range 0 to 48 := 0;               -- Iteration counter
  signal div_active    : std_logic := '0';                         -- Divider busy flag
  signal top_q_nonzero : std_logic := '0';                         -- Overflow detection: top 16 bits

  ----------------------------------------------------------------------------
  -- Output registers
  ----------------------------------------------------------------------------
  signal result       : fp32 := FP_ZERO;                     -- Registered output
  signal result_valid : std_logic := '0';                    -- Valid strobe

  -- Saturation constants for Q16.16
  constant POS_MAX : unsigned(31 downto 0) := x"7FFF_FFFF";  -- +32767.9999847412
  constant NEG_MAX : unsigned(31 downto 0) := x"8000_0000";  -- -32768.0

begin

  main_process : process(clk)
    -- Variables for arithmetic inside the clocked process
    variable diff_val       : signed(31 downto 0);           -- b - y result
    variable a_val          : signed(31 downto 0);           -- divisor
    variable diff_abs       : unsigned(31 downto 0);         -- |b - y|
    variable a_abs          : unsigned(31 downto 0);         -- |a|
    variable temp_remainder : unsigned(32 downto 0);         -- Shifted remainder
    variable new_remainder  : unsigned(32 downto 0);         -- After subtract
    variable qbit           : std_logic;                     -- Current quotient bit

    -- FIFO control
    variable wr_en : boolean;
    variable rd_en : boolean;

    -- Post-division saturation
    variable q_mag_sat : unsigned(31 downto 0);
    variable q_signed  : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- Reset FIFO
        wr_ptr <= 0;
        rd_ptr <= 0;
        count  <= 0;
        -- Reset FSM and working regs
        state        <= IDLE;
        work_a       <= FP_ZERO;
        work_b       <= FP_ZERO;
        work_y       <= FP_ZERO;
        div_sign     <= '0';
        -- Reset divider
        dividend_reg  <= (others => '0');
        divisor_reg   <= (others => '0');
        divisor_ext   <= (others => '0');
        quotient_reg  <= (others => '0');
        remainder_reg <= (others => '0');
        div_counter   <= 0;
        div_active    <= '0';
        top_q_nonzero <= '0';
        -- Reset outputs
        result       <= FP_ZERO;
        result_valid <= '0';

      else
        -- Default: deassert valid (pulse for one cycle only)
        result_valid <= '0';

        ----------------------------------------------------------------------
        -- FIFO write path: Store incoming data if FIFO not full
        ----------------------------------------------------------------------
        wr_en := (enable = '1') and (valid_in = '1') and (count < FIFO_DEPTH);
        if wr_en then
          fifo(wr_ptr).a <= a_in;
          fifo(wr_ptr).b <= b_in;
          fifo(wr_ptr).y <= y_in;
          if wr_ptr = FIFO_DEPTH-1 then
            wr_ptr <= 0;                                     -- Wrap around
          else
            wr_ptr <= wr_ptr + 1;
          end if;
        end if;

        ----------------------------------------------------------------------
        -- FIFO read path: Fetch data when IDLE and divider not busy
        ----------------------------------------------------------------------
        rd_en := (state = IDLE) and (count > 0) and (div_active = '0');
        if rd_en then
          work_a <= fifo(rd_ptr).a;
          work_b <= fifo(rd_ptr).b;
          work_y <= fifo(rd_ptr).y;
          if rd_ptr = FIFO_DEPTH-1 then
            rd_ptr <= 0;                                     -- Wrap around
          else
            rd_ptr <= rd_ptr + 1;
          end if;
        end if;

        -- FIFO count update (handles simultaneous read/write)
        if wr_en and (not rd_en) then
          count <= count + 1;
        elsif rd_en and (not wr_en) then
          count <= count - 1;
        else
          count <= count;  -- Both or neither: unchanged
        end if;

        ----------------------------------------------------------------------
        -- FSM: Control division operation
        ----------------------------------------------------------------------
        case state is

          --------------------------------------------------------------------
          -- IDLE: Wait for data in FIFO
          --------------------------------------------------------------------
          when IDLE =>
            if rd_en then
              state <= PROCESS_DATA;
            else
              state <= IDLE;
            end if;

          --------------------------------------------------------------------
          -- PROCESS_DATA: Setup division (compute diff, get magnitudes)
          --------------------------------------------------------------------
          when PROCESS_DATA =>
            -- Compute signed difference (b - y)
            diff_val := signed(work_b) - signed(work_y);
            a_val    := signed(work_a);

            -- Handle divide-by-zero: return 0
            if a_val = 0 then
              result       <= FP_ZERO;
              result_valid <= '1';
              state        <= IDLE;
            else
              -- Determine sign of final result: XOR of operand signs
              div_sign <= diff_val(31) xor a_val(31);

              -- Get absolute magnitudes for unsigned division
              if diff_val(31) = '1' then
                diff_abs := unsigned(-diff_val);             -- Negate if negative
              else
                diff_abs := unsigned(diff_val);
              end if;

              if a_val(31) = '1' then
                a_abs := unsigned(-a_val);                   -- Negate if negative
              else
                a_abs := unsigned(a_val);
              end if;

              -- Build 48-bit dividend = |b-y| << 16 (scale for Q16.16)
              dividend_reg <= diff_abs & to_unsigned(0, 16);

              -- Load divisor and its 33-bit extension for comparison
              divisor_reg  <= a_abs;
              divisor_ext  <= '0' & a_abs;

              -- Initialize divider state
              quotient_reg  <= (others => '0');
              remainder_reg <= (others => '0');
              div_counter   <= 48;                           -- 48 iterations
              div_active    <= '1';
              top_q_nonzero <= '0';

              state <= DIVIDING;
            end if;

          --------------------------------------------------------------------
          -- DIVIDING: Execute binary division algorithm (1 bit per cycle)
          --------------------------------------------------------------------
          when DIVIDING =>
            if div_counter > 0 then
              -- Step 1: Shift remainder left, bring in next dividend MSB
              temp_remainder := remainder_reg(31 downto 0) & dividend_reg(47);

              -- Step 2: Trial subtraction (compare with divisor)
              if temp_remainder >= divisor_ext then
                -- Remainder >= Divisor: subtract and set quotient bit = 1
                new_remainder := temp_remainder - divisor_ext;
                qbit          := '1';
              else
                -- Remainder < Divisor: keep remainder, quotient bit = 0
                new_remainder := temp_remainder;
                qbit          := '0';
              end if;

              -- Step 3: Update remainder register
              remainder_reg <= new_remainder;

              -- Step 4: Update quotient register
              -- First 16 iterations: detect overflow (integer part > 16 bits)
              -- Last 32 iterations: build the Q16.16 quotient
              if div_counter <= 32 then
                quotient_reg <= quotient_reg(30 downto 0) & qbit;  -- Shift in new bit
              else
                if qbit = '1' then
                  top_q_nonzero <= '1';                       -- Overflow detected
                end if;
                quotient_reg <= quotient_reg;                 -- Keep unchanged
              end if;

              -- Step 5: Shift dividend to expose next bit
              dividend_reg <= dividend_reg(46 downto 0) & '0';

              -- Step 6: Decrement counter
              div_counter <= div_counter - 1;

            else
              -- All 48 bits consumed: division complete
              div_active <= '0';
              state      <= OUTPUT_RESULT;
            end if;

          --------------------------------------------------------------------
          -- OUTPUT_RESULT: Apply sign, saturate, and output
          --------------------------------------------------------------------
          when OUTPUT_RESULT =>
            -- Saturate based on sign and overflow detection
            if div_sign = '0' then
              -- Positive result
              if (top_q_nonzero = '1') or (quotient_reg > POS_MAX) then
                q_mag_sat := POS_MAX;                         -- Clamp to max positive
              else
                q_mag_sat := quotient_reg;
              end if;
            else
              -- Negative result
              if quotient_reg > NEG_MAX then
                q_mag_sat := NEG_MAX;                         -- Clamp to max negative magnitude
              else
                q_mag_sat := quotient_reg;
              end if;
            end if;

            -- Apply sign to get two's complement result
            if div_sign = '1' then
              q_signed := -signed(std_logic_vector(q_mag_sat));
            else
              q_signed :=  signed(std_logic_vector(q_mag_sat));
            end if;

            -- Register outputs
            result       <= q_signed;
            result_valid <= '1';
            state        <= IDLE;

        end case;  -- state

      end if; -- rst
    end if;
  end process;

  -- Output port assignments
  x_out     <= result;
  valid_out <= result_valid;

end architecture simple;