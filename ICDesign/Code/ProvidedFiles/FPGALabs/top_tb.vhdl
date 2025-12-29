-- ############################################################################
-- Simple Band Matrix Solver Testbench - EEE3027 
-- ############################################################################

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.fp_pkg.all;

entity systolic_band_solver_tb is
end entity systolic_band_solver_tb;

architecture true_systolic_verification of systolic_band_solver_tb is

    -- ========================================================================
    -- COMPONENT DECLARATION
    -- ========================================================================
    component systolic_band_solver is
        port (
            clk          : in  std_logic;
            rst          : in  std_logic;
            start_solve  : in  std_logic;
            
            -- 4x4 Band matrix A - only considering diagonal and sub-diagonal elements
            a11, a22, a33, a44 : in  fp32;  -- Diagonal
            a21, a32, a43      : in  fp32;  -- Sub-diagonal
            
            -- Right-hand side vector b
            b1, b2, b3, b4 : in  fp32;
            
            -- Solution vector x
            x1, x2, x3, x4 : out fp32;
            
            -- Status
            solve_done : out std_logic;
            solve_busy : out std_logic
        );
    end component;

    -- ========================================================================
    -- TESTBENCH SIGNALS
    -- ========================================================================
    
    -- Clock and reset
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';
    constant CLK_PERIOD : time := 25 ns;  
    
    -- Control
    signal start_solve : std_logic := '0';
    signal solve_done : std_logic;
    signal solve_busy : std_logic;
    
    -- Matrix A (band matrix - only non-zero elements)
    signal a11, a22, a33, a44 : fp32;  -- Diagonal
    signal a21, a32, a43      : fp32;  -- Sub-diagonal
    
    -- Vectors
    signal b1, b2, b3, b4 : fp32;  -- RHS vector
    signal x1, x2, x3, x4 : fp32;  -- Solution vector
    
    -- ========================================================================
    -- TEST CASES
    -- ========================================================================
    type test_case_t is record
        name : string(1 to 30);
        -- Matrix elements
        a11_val, a22_val, a33_val, a44_val : real;
        a21_val, a32_val, a43_val : real;
        -- RHS vector
        b1_val, b2_val, b3_val, b4_val : real;
        -- Expected solution (for verification)
        x1_exp, x2_exp, x3_exp, x4_exp : real;
    end record;
    
    type test_cases_array is array (0 to 2) of test_case_t;
    
    constant test_cases : test_cases_array := (
        0 => (name => "Identity Matrix Test          ",
              a11_val => 1.0, a22_val => 1.0, a33_val => 1.0, a44_val => 1.0,
              a21_val => 0.0, a32_val => 0.0, a43_val => 0.0,
              b1_val => 1.0, b2_val => 2.0, b3_val => 3.0, b4_val => 4.0,
              x1_exp => 1.0, x2_exp => 2.0, x3_exp => 3.0, x4_exp => 4.0),
              
        1 => (name => "Diagonal and Subdiagonal Test ",
              a11_val => 2.0, a22_val => 2.0, a33_val => 2.0, a44_val => 2.0,
              a21_val => 1.0, a32_val => 1.0, a43_val => 1.0,
              b1_val => 2.0, b2_val => 3.0, b3_val => 3.0, b4_val => 3.0,
              x1_exp => 1.0, x2_exp => 1.0, x3_exp => 1.0, x4_exp => 1.0),
              
        2 => (name => "Textbook Example              ",
              a11_val => 3.0, a22_val => 4.0, a33_val => 2.0, a44_val => 5.0,
              a21_val => -1.0, a32_val => 2.0, a43_val => -1.0,
              b1_val => 6.0, b2_val => 7.0, b3_val => 8.0, b4_val => 8.0,
              x1_exp => 2.0, x2_exp => 2.0, x3_exp => 2.0, x4_exp => 2.0)
    );

begin

    -- ========================================================================
    -- DEVICE UNDER TEST
    -- ========================================================================
    dut: systolic_band_solver
        port map (
            clk          => clk,
            rst          => rst,
            start_solve  => start_solve,
            a11 => a11, a22 => a22, a33 => a33, a44 => a44,
            a21 => a21, a32 => a32, a43 => a43,
            b1 => b1, b2 => b2, b3 => b3, b4 => b4,
            x1 => x1, x2 => x2, x3 => x3, x4 => x4,
            solve_done   => solve_done,
            solve_busy   => solve_busy
        );

    -- ========================================================================
    -- CLOCK GENERATION
    -- ========================================================================
    clock_process: process
    begin
        clk <= '0';
        wait for CLK_PERIOD/2;
        clk <= '1';
        wait for CLK_PERIOD/2;
    end process;

    -- ========================================================================
    -- MAIN TEST PROCESS
    -- ========================================================================
    test_process: process
        
        -- Load test case data
        procedure load_test_case(tc : test_case_t) is
        begin
            -- First reset all inputs to zero
            a11 <= FP_ZERO; a22 <= FP_ZERO; a33 <= FP_ZERO; a44 <= FP_ZERO;
            a21 <= FP_ZERO; a32 <= FP_ZERO; a43 <= FP_ZERO;
            b1 <= FP_ZERO; b2 <= FP_ZERO; b3 <= FP_ZERO; b4 <= FP_ZERO;
            wait for CLK_PERIOD;
            
            -- Load matrix A
            a11 <= real_to_fp(tc.a11_val);
            a22 <= real_to_fp(tc.a22_val);
            a33 <= real_to_fp(tc.a33_val);
            a44 <= real_to_fp(tc.a44_val);
            a21 <= real_to_fp(tc.a21_val);
            a32 <= real_to_fp(tc.a32_val);
            a43 <= real_to_fp(tc.a43_val);
            
            -- Load vector b
            b1 <= real_to_fp(tc.b1_val);
            b2 <= real_to_fp(tc.b2_val);
            b3 <= real_to_fp(tc.b3_val);
            b4 <= real_to_fp(tc.b4_val);
            
            wait for CLK_PERIOD;  -- Extra settling time
        end procedure;
        
        -- Run solver and wait for completion
        procedure run_solver is
            variable timeout : integer := 0;
        begin
            wait until rising_edge(clk);
            -- Make sure we start from a clean state
            start_solve <= '0';
            wait for CLK_PERIOD;
            
            start_solve <= '1';
            wait for CLK_PERIOD;
            start_solve <= '0';
            
            -- Wait for solver to become busy
            timeout := 0;
            while solve_busy = '0' and timeout < 10 loop
                wait for CLK_PERIOD;
                timeout := timeout + 1;
            end loop;
            
            if solve_busy = '1' then
                report "  Systolic array started processing";
            else
                report "  ERROR: Solver did not start!" severity error;
                return;
            end if;
            
            -- Wait for completion (with safety timeout)
            timeout := 0;
            while solve_done = '0' loop
                wait for CLK_PERIOD;
                timeout := timeout + 1;
                if timeout > 2000 then
                    report "  ERROR: Systolic solver timeout!" severity error;
                    exit;
                end if;
            end loop;
            
            if solve_done = '1' then
                report "  Systolic solution completed in " & integer'image(timeout) & " cycles";
            end if;
            
            -- Wait for done signal to clear
            wait for CLK_PERIOD * 5;
        end procedure;
        
        -- Check results
        procedure check_results(tc : test_case_t) is
            variable x1_computed, x2_computed, x3_computed, x4_computed : real;
            variable error1, error2, error3, error4 : real;
            variable max_error : real := 0.01;  -- 1% tolerance
        begin
            -- Convert results back to real
            x1_computed := fp_to_real(x1);
            x2_computed := fp_to_real(x2);
            x3_computed := fp_to_real(x3);
            x4_computed := fp_to_real(x4);
            
            -- Calculate errors
            error1 := abs(x1_computed - tc.x1_exp);
            error2 := abs(x2_computed - tc.x2_exp);
            error3 := abs(x3_computed - tc.x3_exp);
            error4 := abs(x4_computed - tc.x4_exp);
            
            report "=== SOLUTION RESULTS: " & tc.name & " ===";
            report "Matrix A (band form):";
            report "  [" & real'image(tc.a11_val)(1 to 6) & "   0     0     0  ]";
            report "  [" & real'image(tc.a21_val)(1 to 6) & " " & real'image(tc.a22_val)(1 to 6) & "   0     0  ]";
            report "  [ 0   " & real'image(tc.a32_val)(1 to 6) & " " & real'image(tc.a33_val)(1 to 6) & "   0  ]";
            report "  [ 0    0   " & real'image(tc.a43_val)(1 to 6) & " " & real'image(tc.a44_val)(1 to 6) & "]";
            report "";
            report "RHS vector b: [" & real'image(tc.b1_val)(1 to 6) & ", " & 
                   real'image(tc.b2_val)(1 to 6) & ", " & real'image(tc.b3_val)(1 to 6) & ", " & 
                   real'image(tc.b4_val)(1 to 6) & "]";
            report "";
            report "SYSTOLIC ARRAY SOLUTION:";
            report "  x1 = " & real'image(x1_computed)(1 to 8) & " (expected: " & real'image(tc.x1_exp)(1 to 8) & ")";
            report "  x2 = " & real'image(x2_computed)(1 to 8) & " (expected: " & real'image(tc.x2_exp)(1 to 8) & ")";
            report "  x3 = " & real'image(x3_computed)(1 to 8) & " (expected: " & real'image(tc.x3_exp)(1 to 8) & ")";
            report "  x4 = " & real'image(x4_computed)(1 to 8) & " (expected: " & real'image(tc.x4_exp)(1 to 8) & ")";
            report "";
            
            -- Verification
            if error1 < max_error and error2 < max_error and 
               error3 < max_error and error4 < max_error then
                report "PASS: All solutions within tolerance";
            else
                report "FAIL: Solution errors too large" severity warning;
                report "  Errors: " & real'image(error1) & ", " & real'image(error2) & 
                       ", " & real'image(error3) & ", " & real'image(error4);
            end if;
            report "";
        end procedure;

    begin
        -- Reset
        rst <= '1';
        start_solve <= '0';
        
        -- Initialize all input signals
        a11 <= FP_ZERO; a22 <= FP_ZERO; a33 <= FP_ZERO; a44 <= FP_ZERO;
        a21 <= FP_ZERO; a32 <= FP_ZERO; a43 <= FP_ZERO;
        b1 <= FP_ZERO; b2 <= FP_ZERO; b3 <= FP_ZERO; b4 <= FP_ZERO;
        
        wait for CLK_PERIOD * 100;
        rst <= '0';
        wait for CLK_PERIOD * 5;
        
        report "Systolic array reset completed";
        report "";
        
        -- ====================================================================
        -- RUN ALL TEST CASES (with per-case reset preserved)
        -- ====================================================================
        for i in test_cases'range loop
            report "=== TEST CASE " & integer'image(i+1) & " of " & 
                   integer'image(test_cases'length) & " ===";
            
            load_test_case(test_cases(i));
            run_solver;
            check_results(test_cases(i));

            -- keep your resets between examples
            wait for CLK_PERIOD * 10;
            rst <= '1';
            wait for CLK_PERIOD * 50;
            rst <= '0';
            wait for CLK_PERIOD * 5;
        end loop;

        -- Stop after the three examples
        report "All three examples executed. Stopping simulation.";
        std.env.stop;  -- VHDL-2008
        wait;

    end process test_process;
end architecture true_systolic_verification;
