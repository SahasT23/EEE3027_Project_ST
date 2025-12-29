library IEEE;                                                -- Standard libraries
use IEEE.STD_LOGIC_1164.ALL;                                 -- std_logic, std_logic_vector
use IEEE.NUMERIC_STD.ALL;                                    
use work.fp_pkg.all;                                
                                                            

entity divsub is
  port (
    clk       : in  std_logic;                              -- Clock
    rst       : in  std_logic;                              -- Synchronous reset (active '1')
    enable    : in  std_logic;                              -- Enable FIFO write
    a_in      : in  fp32;                                -- Q16.16 divisor (two's complement)
    b_in      : in  fp32;                                -- Q16.16 minuend
    y_in      : in  fp32;                                -- Q16.16 subtrahend
    x_out     : out fp32;                                -- Q16.16 result: (b_in - y_in) / a_in
    valid_in  : in  std_logic;                              
    valid_out : out std_logic                               
  );
end entity divsub;

architecture simple of divsub is

  ----------------------------------------------------------------------------
  -- Simple FIFO to decouple input from divider
  ----------------------------------------------------------------------------
  type fifo_data_t is record
    a : fp32;
    b : fp32;
    y : fp32;
  end record;

  constant FIFO_DEPTH : integer := 32;                       -- Small queue
  type fifo_array_t is array (0 to FIFO_DEPTH-1) of fifo_data_t;
  signal fifo   : fifo_array_t;
  signal wr_ptr : integer range 0 to FIFO_DEPTH-1 := 0;
  signal rd_ptr : integer range 0 to FIFO_DEPTH-1 := 0;
  signal count  : integer range 0 to FIFO_DEPTH := 0;

  ----------------------------------------------------------------------------
  -- Control FSM
  ----------------------------------------------------------------------------
  type state_t is (IDLE, PROCESS_DATA, DIVIDING, OUTPUT_RESULT);
  signal state : state_t := IDLE;

  ----------------------------------------------------------------------------
  -- Latched work registers
  ----------------------------------------------------------------------------
  signal work_a, work_b, work_y : fp32 := FP_ZERO;          
  signal div_sign : std_logic := '0';                          -- sign((b-y)/a)

  ----------------------------------------------------------------------------
  -- Divider internals (unsigned magnitude division; sign applied at the end)
  -- We build a 48-bit dividend = |b-y| << 16 to preserve Q16.16 fractional bits
  ----------------------------------------------------------------------------
  signal dividend_reg  : unsigned(47 downto 0) := (others => '0'); -- 48-bit dividend
  signal divisor_reg   : unsigned(31 downto 0) := (others => '0'); -- 32-bit divisor magnitude
  signal divisor_ext   : unsigned(32 downto 0) := (others => '0'); -- 33-bit for compare/sub
  signal quotient_reg  : unsigned(31 downto 0) := (others => '0'); -- 32-bit Q16.16 quotient magnitude
  signal remainder_reg : unsigned(32 downto 0) := (others => '0'); -- 33-bit remainder (guard bit)
  signal div_counter   : integer range 0 to 48 := 0;               
  signal div_active    : std_logic := '0';                         -- Divider busy flag
  signal top_q_nonzero : std_logic := '0';                         -- Any of the top 16 quotient bits set (overflow of integer field)

  ----------------------------------------------------------------------------
  -- Output regs and constants
  ----------------------------------------------------------------------------
  signal result       : fp32 := FP_ZERO;                     -- Registered output
  signal result_valid : std_logic := '0';                        -- Valid strobe

  constant POS_MAX : unsigned(31 downto 0) := x"7FFF_FFFF";     -- +32767.9999847412 in Q16.16
  constant NEG_MAX : unsigned(31 downto 0) := x"8000_0000";     -- represents -32768.0 in Q16.16

begin

  main_process : process(clk)
    -- Variables for arithmetic inside the clocked process
    variable diff_val       : signed(31 downto 0);
    variable a_val          : signed(31 downto 0);
    variable diff_abs       : unsigned(31 downto 0);
    variable a_abs          : unsigned(31 downto 0);
    variable temp_remainder : unsigned(32 downto 0);
    variable new_remainder  : unsigned(32 downto 0);
    variable qbit           : std_logic;

    -- FIFO control
    variable wr_en : boolean;
    variable rd_en : boolean;

    -- Post-division pack
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
        -- Default deassert valid
        result_valid <= '0';

        ----------------------------------------------------------------------
        -- FIFO write path
        ----------------------------------------------------------------------
        wr_en := (enable = '1') and (valid_in = '1') and (count < FIFO_DEPTH);
        if wr_en then
          fifo(wr_ptr).a <= a_in;
          fifo(wr_ptr).b <= b_in;
          fifo(wr_ptr).y <= y_in;
          if wr_ptr = FIFO_DEPTH-1 then
            wr_ptr <= 0;
          else
            wr_ptr <= wr_ptr + 1;
          end if;
        end if;

        ----------------------------------------------------------------------
        -- FIFO read path (only when idle and divider not busy)
        ----------------------------------------------------------------------
        rd_en := (state = IDLE) and (count > 0) and (div_active = '0');
        if rd_en then
          work_a <= fifo(rd_ptr).a;
          work_b <= fifo(rd_ptr).b;
          work_y <= fifo(rd_ptr).y;
          if rd_ptr = FIFO_DEPTH-1 then
            rd_ptr <= 0;
          else
            rd_ptr <= rd_ptr + 1;
          end if;
        end if;

        -- FIFO count update (atomic)
        if wr_en and (not rd_en) then
          count <= count + 1;
        elsif rd_en and (not wr_en) then
          count <= count - 1;
        else
          count <= count;  -- both or neither → unchanged
        end if;

        ----------------------------------------------------------------------
        -- FSM
        ----------------------------------------------------------------------
        case state is

          --------------------------------------------------------------------
          when IDLE =>
            if rd_en then
              state <= PROCESS_DATA;
            else
              state <= IDLE;
            end if;

          --------------------------------------------------------------------
          when PROCESS_DATA =>
            -- Compute signed difference (b - y)
            diff_val := signed(work_b) - signed(work_y);
            a_val    := signed(work_a);

            -- Early divide-by-zero handling
            if a_val = 0 then
              result       <= FP_ZERO;   -- define as 0 (or choose a policy)
              result_valid <= '1';
              state        <= IDLE;
            else
              -- Determine sign of final result
              div_sign <= diff_val(31) xor a_val(31);

              -- Absolute magnitudes
              if diff_val(31) = '1' then
                diff_abs := unsigned(-diff_val);
              else
                diff_abs := unsigned(diff_val);
              end if;

              if a_val(31) = '1' then
                a_abs := unsigned(-a_val);
              else
                a_abs := unsigned(a_val);
              end if;

              -- Build 48-bit dividend = |b-y| << 16 (append 16 zero LSBs)
              dividend_reg <= diff_abs & to_unsigned(0, 16);

              -- Load divisor and its 33-bit extension
              divisor_reg  <= a_abs;
              divisor_ext  <= '0' & a_abs;

              -- Reset divider machine
              quotient_reg  <= (others => '0');
              remainder_reg <= (others => '0');
              div_counter   <= 48;          -- process all 48 dividend bits
              div_active    <= '1';
              top_q_nonzero <= '0';

              state <= DIVIDING;
            end if;

          --------------------------------------------------------------------
          when --what state is this? =>
            if div_counter > 0 then
              -- Shift remainder left by 1 and bring in the next dividend MSB
              temp_remainder := remainder_reg(31 downto 0) & dividend_reg(47);

              -- Trial subtraction
              if temp_remainder >= divisor_ext then
                new_remainder := temp_remainder - --what do you need to take away here?;
                qbit          := '1';
              else
                new_remainder := --what goes here?;
                qbit          := '0';
              end if;

              -- Commit new remainder
              remainder_reg <= --what goes here?;

              -- We run 48 cycles: first 16 produce the top quotient bits we ignore
              -- for output (but track to detect positive overflow), last 32 fill quotient_reg
              if div_counter <= 32 then
                quotient_reg <= quotient_reg(30 downto 0) & qbit;  -- keep lower 32 bits (Q16.16)
              else
                if qbit = '1' then
                  top_q_nonzero <= '1';                           -- remember integer overflow beyond 16 bits
                end if;
                quotient_reg <= --what goes here?;                      -- unchanged
              end if;

              -- Shift dividend to expose next bit
              dividend_reg <= dividend_reg(46 downto 0) & '0';

              -- Next iteration
              div_counter <= --what goes here?;

            else
              -- All bits consumed
              div_active <= '0';
              state      <= OUTPUT_RESULT;
            end if;

          --------------------------------------------------------------------
          when OUTPUT_RESULT =>
            -- Saturate magnitude according to sign
            if div_sign = '0' then
              if (top_q_nonzero = '1') or (quotient_reg > POS_MAX) then
                q_mag_sat := POS_MAX;           -- positive overflow → clamp
              else
                q_mag_sat := quotient_reg;      -- within range
              end if;
            else
              if quotient_reg > NEG_MAX then
                q_mag_sat := NEG_MAX;           -- negative magnitude too large → clamp to -32768.0
              else
                q_mag_sat := quotient_reg;
              end if;
            end if;

            -- Apply sign over full 32-bit width
            if div_sign = '1' then
              q_signed := -signed(std_logic_vector(q_mag_sat));
            else
              q_signed :=  signed(std_logic_vector(q_mag_sat));
            end if;

            -- Register outputs
            result       <=  q_signed;
            result_valid <= '1';
            state        <= IDLE;

        end case;  -- state

      end if; -- rst
    end if;
  end process;

  -- Output ports
  x_out     <= result;
  valid_out <= result_valid;

end architecture simple;
