--------------------------------------------------------------------------------
-- controller.vhdl - Systolic Array Controller (Systolic Band Solver)
-- EEE3027 IC Design Labs - Lab 6
--
-- This controller orchestrates a simplified systolic array for solving
-- banded linear systems. It connects:
--   - 1 x DivSub block (boundary cell): computes x = (b - y) / a
--   - 1 x IPSP block (internal cell): computes y = a*x + y_in
--
-- The controller manages:
--   - Input loading from testbench
--   - Sequencing of DivSub (multi-cycle) and IPSP (pipelined)
--   - Output coordination
--
-- Architecture Overview:
--   b[i] → DivSub → x[i] → IPSP → y_partial
--            ↑                ↓
--          a[i]            Accumulate
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.fp_pkg.all;

entity systolic_band_solver is
  port (
    clk        : in  std_logic;
    rst        : in  std_logic;
    en         : in  std_logic;

    -- Input interface (from testbench)
    start      : in  std_logic;           -- Start computation pulse
    a_divsub   : in  fp32;                -- Diagonal coefficient for DivSub
    b_in       : in  fp32;                -- RHS value b
    y_feedback : in  fp32;                -- Feedback y for DivSub
    a_ipsp     : in  fp32;                -- Coefficient for IPSP
    y_ipsp_in  : in  fp32;                -- Partial sum input for IPSP

    -- Output interface
    done       : out std_logic;           -- Computation complete
    x_result   : out fp32;                -- Solved x value
    y_result   : out fp32                 -- IPSP output (for chaining)
  );
end entity;

architecture rtl of systolic_band_solver is

  ----------------------------------------------------------------------------
  -- FSM States
  ----------------------------------------------------------------------------
  type state_type is (
    S_IDLE,           -- Waiting for start signal
    S_DIVSUB_WAIT,    -- Waiting for DivSub to complete (~50 cycles)
    S_IPSP_WAIT,      -- Waiting for IPSP pipeline to flush (4 cycles)
    S_DONE            -- Output results
  );
  signal state : state_type := S_IDLE;

  ----------------------------------------------------------------------------
  -- Internal Signals
  ----------------------------------------------------------------------------
  -- DivSub interface
  signal divsub_valid_in  : std_logic := '0';
  signal divsub_valid_out : std_logic;
  signal divsub_a         : fp32;
  signal divsub_b         : fp32;
  signal divsub_y         : fp32;
  signal divsub_x_out     : fp32;

  -- IPSP interface
  signal ipsp_valid_in    : std_logic := '0';
  signal ipsp_valid_out   : std_logic;
  signal ipsp_a           : fp32;
  signal ipsp_x           : fp32;
  signal ipsp_y_in        : fp32;
  signal ipsp_y_out       : fp32;
  signal ipsp_a_out       : fp32;  -- Pass-through (unused but connected)
  signal ipsp_x_out       : fp32;  -- Pass-through (unused but connected)

  -- Pipeline counter for IPSP flush
  signal ipsp_wait_count : integer range 0 to 10 := 0;

  -- Stored inputs (latched on start)
  signal stored_a_ipsp   : fp32;
  signal stored_y_ipsp   : fp32;

begin

  ----------------------------------------------------------------------------
  -- DivSub Instance: Computes x = (b - y) / a
  -- Multi-cycle: ~50 clock cycles latency
  ----------------------------------------------------------------------------
  divsub_inst: entity work.divsub
    port map (
      clk       => clk,
      rst       => rst,
      en        => en,
      valid_in  => divsub_valid_in,
      a_in      => divsub_a,
      b_in      => divsub_b,
      y_in      => divsub_y,
      valid_out => divsub_valid_out,
      x_out     => divsub_x_out
    );

  ----------------------------------------------------------------------------
  -- IPSP Instance: Computes y = a*x + y_in
  -- Pipelined: 4 clock cycles latency
  ----------------------------------------------------------------------------
  ipsp_inst: entity work.ipsp_pipelined
    port map (
      clk       => clk,
      rst       => rst,
      en        => en,
      valid_in  => ipsp_valid_in,
      a_in      => ipsp_a,
      x_in      => ipsp_x,
      y_in      => ipsp_y_in,
      valid_out => ipsp_valid_out,
      a_out     => ipsp_a_out,
      x_out     => ipsp_x_out,
      y_out     => ipsp_y_out
    );

  ----------------------------------------------------------------------------
  -- Main Controller FSM
  ----------------------------------------------------------------------------
  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        state            <= S_IDLE;
        done             <= '0';
        x_result         <= (others => '0');
        y_result         <= (others => '0');
        divsub_valid_in  <= '0';
        ipsp_valid_in    <= '0';
        ipsp_wait_count  <= 0;
        divsub_a         <= (others => '0');
        divsub_b         <= (others => '0');
        divsub_y         <= (others => '0');
        ipsp_a           <= (others => '0');
        ipsp_x           <= (others => '0');
        ipsp_y_in        <= (others => '0');
        stored_a_ipsp    <= (others => '0');
        stored_y_ipsp    <= (others => '0');

      elsif en = '1' then
        case state is

          ------------------------------------------------------------------
          -- S_IDLE: Wait for start signal, latch inputs
          ------------------------------------------------------------------
          when S_IDLE =>
            done <= '0';
            
            if start = '1' then
              -- Latch all inputs
              divsub_a      <= a_divsub;     -- Diagonal element
              divsub_b      <= b_in;         -- RHS value
              divsub_y      <= y_feedback;   -- Feedback from previous row
              stored_a_ipsp <= a_ipsp;       -- Store for IPSP
              stored_y_ipsp <= y_ipsp_in;    -- Store for IPSP
              
              -- Trigger DivSub computation
              divsub_valid_in <= '1';
              state           <= S_DIVSUB_WAIT;
            end if;

          ------------------------------------------------------------------
          -- S_DIVSUB_WAIT: Wait for DivSub to complete
          -- DivSub takes ~50 cycles for multi-cycle division
          ------------------------------------------------------------------
          when S_DIVSUB_WAIT =>
            divsub_valid_in <= '0';  -- Clear trigger
            
            if divsub_valid_out = '1' then
              -- DivSub complete, x value is ready
              -- Store x result
              x_result <= divsub_x_out;
              
              -- Feed x into IPSP: y_out = a * x + y_in
              ipsp_a       <= stored_a_ipsp;
              ipsp_x       <= divsub_x_out;   -- x from DivSub
              ipsp_y_in    <= stored_y_ipsp;
              ipsp_valid_in <= '1';
              
              ipsp_wait_count <= 0;
              state           <= S_IPSP_WAIT;
            end if;

          ------------------------------------------------------------------
          -- S_IPSP_WAIT: Wait for IPSP pipeline to produce output
          -- IPSP has 4-cycle latency
          ------------------------------------------------------------------
          when S_IPSP_WAIT =>
            ipsp_valid_in <= '0';  -- Clear trigger
            
            if ipsp_valid_out = '1' then
              -- IPSP complete
              y_result <= ipsp_y_out;
              state    <= S_DONE;
            else
              -- Safety timeout (shouldn't be needed)
              ipsp_wait_count <= ipsp_wait_count + 1;
              if ipsp_wait_count > 8 then
                state <= S_DONE;  -- Timeout failsafe
              end if;
            end if;

          ------------------------------------------------------------------
          -- S_DONE: Signal completion
          ------------------------------------------------------------------
          when S_DONE =>
            done  <= '1';
            state <= S_IDLE;

          when others =>
            state <= S_IDLE;

        end case;
      end if;
    end if;
  end process;

end architecture;
