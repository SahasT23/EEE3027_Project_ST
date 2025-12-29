library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.fp_pkg.all; 

entity systolic_band_solver is
    port (
        clk          : in  std_logic;
        rst          : in  std_logic;   -- external async reset; synchronized inside
        start_solve  : in  std_logic;
        
        -- Matrix inputs (diagonal D and subdiagonal L)
        a11, a22, a33, a44 : in  fp32; -- diagonal
        a21, a32, a43      : in  fp32; -- subdiagonal (below diagonal)
        
        -- RHS vector
        b1, b2, b3, b4 : in  fp32;
        
        -- Solution outputs
        x1, x2, x3, x4 : out fp32;
        
        -- Status
        solve_done : out std_logic;
        solve_busy : out std_logic
    );
end systolic_band_solver;

architecture sutdent_template of systolic_band_solver is

    --------------------------------------------------------------------------
    -- Subcomponents
    --------------------------------------------------------------------------
    component divsub
        port (
            clk       : in  std_logic;
            rst       : in  std_logic;
            enable    : in  std_logic;
            a_in      : in  fp32;
            b_in      : in  fp32;
            y_in      : in  fp32;
            x_out     : out fp32;
            valid_in  : in  std_logic;
            valid_out : out std_logic
        );
    end component;
    
    component ipsp_pipelined
        port (
            clk         : in  std_logic;
            rst         : in  std_logic;
            en          : in  std_logic;
            a_in        : in  fp32;
            x_in        : in  fp32;
            y_in        : in  fp32;
            a_out       : out fp32;
            x_out       : out fp32;
            y_out       : out fp32;
            valid_in    : in  std_logic;
            valid_out   : out std_logic
        );
    end component;

    --------------------------------------------------------------------------
    -- Reset synchronizer (2-FF) and synchronous warmup
    --------------------------------------------------------------------------
    attribute ASYNC_REG : string;

    signal rst_meta     : std_logic := '1';
    signal rst_syncd    : std_logic := '1';
    attribute ASYNC_REG of rst_meta  : signal is "TRUE";
    attribute ASYNC_REG of rst_syncd : signal is "TRUE";

    signal rst_int      : std_logic := '1';  -- fully synchronous internal reset
    signal warmup       : unsigned(3 downto 0) := (others => '0');
    signal warmup_done  : std_logic := '0';

    --------------------------------------------------------------------------
    -- Systolic interconnect
    --------------------------------------------------------------------------
    signal y_flow_left   : fp32;  -- legacy name used elsewhere
    signal x_flow_right  : fp32;

    type fp_array_4 is array (0 to 3) of fp32;
    signal a_flow_down   : fp_array_4 := (others => FP_ZERO);  -- kept for compat

    --------------------------------------------------------------------------
    -- Control signals to blocks
    --------------------------------------------------------------------------
    signal divsub_enable    : std_logic := '0';
    signal divsub_valid_in  : std_logic := '0';
    signal divsub_valid_out : std_logic;
    signal divsub_a_in      : fp32 := FP_ZERO;
    signal divsub_b_in      : fp32 := FP_ZERO;
    
    signal ipsp_enable     : std_logic := '0';
    signal ipsp_valid_in   : std_logic := '0';
    signal ipsp_valid_out  : std_logic;

    -- Gated (masked) enables/valids during warmup
    signal divsub_enable_g : std_logic;
    signal ipsp_enable_g   : std_logic;
    signal ipsp_valid_in_g : std_logic;

    --------------------------------------------------------------------------
    -- Local storage for matrix and RHS
    --------------------------------------------------------------------------
    type fp_diag_array is array (1 to 4) of fp32;
    type fp_sub_array  is array (1 to 3) of fp32;
    type fp_rhs_array  is array (1 to 4) of fp32;
    
    signal diagonal_elements : fp_diag_array := (others => FP_ZERO);
    signal subdiag_elements  : fp_sub_array  := (others => FP_ZERO);
    signal rhs_elements      : fp_rhs_array  := (others => FP_ZERO);
    
    --------------------------------------------------------------------------
    -- Captured solutions
    --------------------------------------------------------------------------
    signal x1_captured, x2_captured, x3_captured, x4_captured : fp32 := FP_ZERO;
    
    --------------------------------------------------------------------------
    -- Controller FSM
    --------------------------------------------------------------------------
    type state_t is (IDLE, LOAD_DATA, SYSTOLIC_PIPELINE, DONE);
    signal state : state_t := IDLE;
    signal cycle_counter  : integer range 0 to 255 := 0;
    signal pipeline_stage : integer range 0 to 7   := 0;
    signal stage_complete : std_logic_vector(1 to 7) := (others => '0');
    
    signal y_flow_left_1: fp32 := FP_ZERO;

    --------------------------------------------------------------------------
    -- Robust y handling for DIVSUB
    --------------------------------------------------------------------------
    signal y_flow_acc  : fp32 := FP_ZERO;  -- registered y used by divsub
    signal y_flow_next : fp32 := FP_ZERO;  -- raw y from ipsp

    --------------------------------------------------------------------------
    -- Stable IPSP command regs (loaded 1 clk before valid pulse)
    --------------------------------------------------------------------------
    signal ipsp_a_cmd  : fp32 := FP_ZERO;
    signal ipsp_x_cmd  : fp32 := FP_ZERO;

begin
    ----------------------------------------------------------------------------
    -- Reset sync (no async PRE/CLR inside design) and synchronous warmup
    ----------------------------------------------------------------------------
    process(clk)
    begin
        if rising_edge(clk) then
            rst_meta  <= rst;       -- sample external reset
            rst_syncd <= rst_meta;  -- second stage
        end if;
    end process;
    rst_int <= rst_syncd;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst_int = '1' then
                warmup      <= (others => '0');
                warmup_done <= '0';
            else
                if warmup_done = '0' then
                    warmup <= warmup + 1;
                    if warmup = "0101" then    -- 5 quiet cycles after reset
                        warmup_done <= '1';
                    end if;
                end if;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Instances
    ----------------------------------------------------------------------------
    divsub_proc: divsub
        port map (
            clk       => clk,
            rst       => rst_int,
            enable    => divsub_enable_g,
            a_in      => divsub_a_in,
            b_in      => divsub_b_in,
            y_in      => y_flow_acc,     -- registered; never 'X'
            x_out     => x_flow_right,
            valid_in  => divsub_valid_in,
            valid_out => divsub_valid_out
        );

    ipsp_proc: ipsp_pipelined
        port map (
            clk         => clk,
            rst         => rst_int,
            en          => ipsp_enable_g,
            a_in        => ipsp_a_cmd,       -- stable cmd regs
            x_in        => ipsp_x_cmd,       -- stable cmd regs
            y_in        => y_flow_left_1,    -- zero (no accumulate inside IPSP)
            a_out       => open,   
            x_out       => open,      
            y_out       => y_flow_next,      -- captured into y_flow_acc on valid_out
            valid_in    => ipsp_valid_in_g,
            valid_out   => ipsp_valid_out
        );

    -- gate enables/valid during warmup to avoid DSP/BRAM timing notifiers
    divsub_enable_g <= divsub_enable and warmup_done;
    ipsp_enable_g   <= ipsp_enable   and warmup_done;
    ipsp_valid_in_g <= ipsp_valid_in and warmup_done;

    -- legacy name drive
    y_flow_left   <= y_flow_acc;
    y_flow_left_1 <= FP_ZERO;

    -- Registered latch for y from IPSP (only when valid_out = '1')
    y_latch: process(clk)
    begin
        if rising_edge(clk) then
            if rst_int = '1' then
                y_flow_acc <= FP_ZERO;
            else
                if ipsp_valid_out = '1' then
                    y_flow_acc <= y_flow_next;
                end if;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Stage completion detector (synchronous reset)
    ----------------------------------------------------------------------------
    stage_coordinator: process(clk)
    begin
        if rising_edge(clk) then
            if rst_int = '1' then
                stage_complete <= (others => '0');
            else
                case pipeline_stage is
                    when 1 => stage_complete(1) <= divsub_valid_out;
                    when 2 => stage_complete(2) <= ipsp_valid_out;
                    when 3 => stage_complete(3) <= divsub_valid_out;
                    when 4 => stage_complete(4) <= ipsp_valid_out;
                    when 5 => stage_complete(5) <= divsub_valid_out;
                    when 6 => stage_complete(6) <= ipsp_valid_out;
                    when 7 => stage_complete(7) <= divsub_valid_out;
                    when others => stage_complete <= (others => '0');
                end case;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Controller FSM with IPSP cmd regs & one-cycle valid pulses (sync reset)
    ----------------------------------------------------------------------------
    controller: process(clk)
    begin
        if rising_edge(clk) then
            if rst_int = '1' then
                -- Clear stored data
                diagonal_elements <= (others => FP_ZERO);
                subdiag_elements  <= (others => FP_ZERO);
                rhs_elements      <= (others => FP_ZERO);

                x1_captured <= FP_ZERO;
                x2_captured <= FP_ZERO;
                x3_captured <= FP_ZERO;
                x4_captured <= FP_ZERO;

                a_flow_down <= (others => FP_ZERO);

                state <= IDLE;
                cycle_counter  <= 0;
                pipeline_stage <= 0;

                divsub_enable   <= '0';
                divsub_valid_in <= '0';
                divsub_a_in     <= FP_ZERO;
                divsub_b_in     <= FP_ZERO;

                ipsp_enable     <= '0';
                ipsp_valid_in   <= '0';
                ipsp_a_cmd      <= FP_ZERO;
                ipsp_x_cmd      <= FP_ZERO;

            else
                -- one-shots default low each cycle
                divsub_valid_in <= '0';
                ipsp_valid_in   <= '0';
                -- NOTE: do NOT zero ipsp_a_cmd/ipsp_x_cmd here

                case state is

                    when IDLE =>
                        cycle_counter  <= 0;
                        pipeline_stage <= 0;
                        a_flow_down    <= (others => FP_ZERO);
                        ipsp_enable    <= '0';
                        divsub_enable  <= '0';
                        if start_solve = '1' then
                            state <= LOAD_DATA;
                        end if;

                    when LOAD_DATA =>
                        -- Load static data (registered copies)
                        diagonal_elements(1) <= a11; diagonal_elements(2) <= a22;
                        diagonal_elements(3) <= a33; diagonal_elements(4) <= a44;
                        subdiag_elements(1)  <= a21; subdiag_elements(2)  <= a32; 
                        subdiag_elements(3)  <= a43;
                        rhs_elements(1)      <= b1;  rhs_elements(2)      <= b2;
                        rhs_elements(3)      <= b3;  rhs_elements(4)      <= b4;

                        pipeline_stage <= 1;
                        state          <= SYSTOLIC_PIPELINE;
                        cycle_counter  <= 0;

                    when SYSTOLIC_PIPELINE =>
                        case pipeline_stage is

                            ------------------------------------------------------------------
                            -- Stage 1: x1 = b1 / a11  (DIVSUB)
                            ------------------------------------------------------------------
                            when 1 =>
                                if cycle_counter = 0 then
                                    divsub_enable   <= What goes here?
                                    divsub_a_in     <= diagonal_elements(1);
                                    divsub_b_in     <= rhs_elements(1);
                                    cycle_counter   <= What goes here?
                                elsif cycle_counter = 1 then
                                    divsub_valid_in <= What goes here?
                                    cycle_counter   <= What goes here?
                                elsif cycle_counter = 2 then
                                    cycle_counter   <= 3;
                                elsif divsub_valid_out = '1' then
                                    divsub_enable   <= What goes here?
                                    x1_captured     <= x_flow_right;
                                    pipeline_stage  <= What goes here?
                                    cycle_counter   <= 0;
                                else
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 2: y = a21 * x1   (IPSP)
                            ------------------------------------------------------------------
                            when 2 =>
                                if cycle_counter = 0 then
                                    ipsp_enable   <= '1';
                                    ipsp_a_cmd    <= subdiag_elements(1);  -- a21
                                    ipsp_x_cmd    <= x1_captured;          -- captured x1
                                    cycle_counter <= What goes here?
                                elsif cycle_counter = 1 then
                                    ipsp_enable   <= What goes here?
                                    ipsp_valid_in <= What goes here?
                                    cycle_counter <= What goes here?
                                elsif cycle_counter = 2 then
                                    ipsp_enable   <= '1';
                                    cycle_counter <= 3;
                                elsif ipsp_valid_out = '1' then
                                    ipsp_enable    <= '0';
                                    pipeline_stage <= What goes here?
                                    cycle_counter  <= 0;
                                else
                                    ipsp_enable   <= '1';
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 3: x2 = (b2 - y) / a22  (DIVSUB)
                            ------------------------------------------------------------------
                            when 3 =>
                                if cycle_counter = 0 then
                                    divsub_enable   <= What goes here?
                                    divsub_a_in     <= diagonal_elements(2);
                                    divsub_b_in     <= rhs_elements(2);
                                    cycle_counter   <= 1;
                                elsif cycle_counter = 1 then
                                    divsub_valid_in <= What goes here?
                                    cycle_counter   <= 2;
                                elsif cycle_counter = 2 then
                                    cycle_counter   <= 3;
                                elsif divsub_valid_out = '1' then
                                    divsub_enable   <= What goes here?
                                    x2_captured     <= x_flow_right;
                                    pipeline_stage  <= What goes here?
                                    cycle_counter   <= 0;
                                else
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 4: y = a32 * x2   (IPSP)
                            ------------------------------------------------------------------
                            when 4 =>
                                if cycle_counter = 0 then
                                    ipsp_enable   <= What goes here?
                                    ipsp_a_cmd    <= subdiag_elements(2);  -- a32
                                    ipsp_x_cmd    <= x2_captured;          -- captured x2
                                    cycle_counter <= 1;
                                elsif cycle_counter = 1 then
                                    ipsp_enable   <= What goes here?
                                    ipsp_valid_in <= What goes here?
                                    cycle_counter <= 2;
                                elsif cycle_counter = 2 then
                                    ipsp_enable   <= '1';
                                    cycle_counter <= 3;
                                elsif ipsp_valid_out = '1' then
                                    ipsp_enable    <= '0';
                                    pipeline_stage <= What goes here?
                                    cycle_counter  <= 0;
                                else
                                    ipsp_enable   <= '1';
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 5: x3 = (b3 - y) / a33  (DIVSUB)
                            ------------------------------------------------------------------
                            when 5 =>
                                if cycle_counter = 0 then
                                    divsub_enable   <= '1';
                                    divsub_a_in     <= diagonal_elements(3);
                                    divsub_b_in     <= rhs_elements(3);
                                    cycle_counter   <= 1;
                                elsif cycle_counter = 1 then
                                    divsub_valid_in <= '1';
                                    cycle_counter   <= 2;
                                elsif cycle_counter = 2 then
                                    cycle_counter   <= 3;
                                elsif divsub_valid_out = '1' then
                                    divsub_enable   <= '0';
                                    x3_captured     <= x_flow_right;
                                    pipeline_stage  <= What goes here?
                                    cycle_counter   <= 0;
                                else
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 6: y = a43 * x3   (IPSP)
                            ------------------------------------------------------------------
                            when 6 =>
                                if cycle_counter = 0 then
                                    ipsp_enable   <= '1';
                                    ipsp_a_cmd    <= subdiag_elements(3);  -- a43
                                    ipsp_x_cmd    <= x3_captured;          -- captured x3
                                    cycle_counter <= 1;
                                elsif cycle_counter = 1 then
                                    ipsp_enable   <= What goes here?
                                    ipsp_valid_in <= What goes here?
                                    cycle_counter <= 2;
                                elsif cycle_counter = 2 then
                                    ipsp_enable   <= What goes here?
                                    cycle_counter <= 3;
                                elsif ipsp_valid_out = '1' then
                                    ipsp_enable    <= '0';
                                    pipeline_stage <= What goes here?
                                    cycle_counter  <= 0;
                                else
                                    ipsp_enable   <= '1';
                                    cycle_counter <= What goes here?
                                end if;

                            ------------------------------------------------------------------
                            -- Stage 7: x4 = (b4 - y) / a44  (DIVSUB)
                            ------------------------------------------------------------------
                            when 7 =>
                                if cycle_counter = 0 then
                                    divsub_enable   <= What goes here?
                                    divsub_a_in     <= diagonal_elements(4);
                                    divsub_b_in     <= rhs_elements(4);
                                    cycle_counter   <= What goes here?
                                elsif cycle_counter = 1 then
                                    divsub_valid_in <= What goes here?
                                    cycle_counter   <= 2;
                                elsif cycle_counter = 2 then
                                    cycle_counter   <= 3;
                                elsif divsub_valid_out = '1' then
                                    divsub_enable   <= What goes here?
                                    x4_captured     <= x_flow_right;
                                    pipeline_stage  <= 0;
                                    cycle_counter   <= 0;
                                    state           <= DONE;
                                else
                                    cycle_counter <= What goes here?
                                end if;

                            when others =>
                                state <= DONE;
                                cycle_counter <= What goes here?
                        end case;

                    when DONE =>
                        pipeline_stage <= 0;
                        if cycle_counter = 0 then
                            cycle_counter <=What goes here?
                        else
                            state <= IDLE;
                            cycle_counter <= What goes here?
                        end if;
                        
                    when others =>
                        state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Outputs
    ----------------------------------------------------------------------------
    x1 <= x1_captured;
    x2 <= x2_captured;
    x3 <= x3_captured;
    x4 <= x4_captured;

    solve_busy <= '1' when state /= IDLE and state /= DONE else '0';
    solve_done <= '1' when state = DONE else '0';

end sutdent_template;
