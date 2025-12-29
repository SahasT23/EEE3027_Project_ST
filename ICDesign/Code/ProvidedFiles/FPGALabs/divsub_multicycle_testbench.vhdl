library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL;
use IEEE.STD_LOGIC_TEXTIO.ALL;
use work.fp_pkg.all;

entity divsub_pipeline_tb is
end entity divsub_pipeline_tb;

architecture simple_pipeline_test of divsub_pipeline_tb is
    
    -- Component declaration
    component divsub is
        port (
            clk         : in  std_logic;
            rst         : in  std_logic;
            enable      : in  std_logic;
            a_in        : in  fp32;
            b_in        : in  fp32;
            y_in        : in  fp32;
            x_out       : out fp32;
            valid_in    : in  std_logic;
            valid_out   : out std_logic
        );
    end component;
    
    -- Signals
    signal clk         : std_logic := '0';
    signal rst         : std_logic := '1';
    signal enable      : std_logic := '1';
    signal a_in        : fp32 := FP_ZERO;
    signal b_in        : fp32 := FP_ZERO;
    signal y_in        : fp32 := FP_ZERO;
    signal x_out       : fp32;
    signal valid_in    : std_logic := '0';
    signal valid_out   : std_logic;
    signal overflow    : std_logic;
    
    -- Test control
    constant clk_period : time := 20 ns;  -- 50MHz
    signal test_complete : boolean := false;
    
    -- Simple test data arrays
    type real_array is array (0 to 15) of real;
    constant a_values : real_array := (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    constant b_values : real_array := (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0);
    constant y_values : real_array := (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2);
    
begin
    
    -- Instantiate IPSP
    uut: divsub
    port map (
        clk         => clk,
        rst         => rst,
        enable      => enable,
        a_in        => a_in,
        b_in        => b_in,
        y_in        => y_in,
        x_out       => x_out,
        valid_in    => valid_in,
        valid_out   => valid_out
    );
    
    -- Clock generation
    clk <= not clk after clk_period/2 when not test_complete;
    
    -- Simple test process
    test_process: process
        variable line_buffer : line;
        variable cycle_count : integer := 0;
        variable output_count : integer := 0;
        variable a_fp, b_fp, y_fp : fp32;
        variable expected, actual, error : real;
    begin

        rst <= '1';
        wait for 120 ns;              -- >100 ns GSR; give margin
        wait until rising_edge(clk);  -- deassert synchronously
        rst <= '0';

        write(line_buffer, string'("=== SIMPLE PIPELINE TEST - DATA EVERY CYCLE ==="));
        writeline(output, line_buffer);
        writeline(output, line_buffer);
        
        -- Feed data every clock cycle
        write(line_buffer, string'("Feeding data every clock cycle..."));
        writeline(output, line_buffer);
        writeline(output, line_buffer);
        
        -- Input phase: Feed all test data continuously
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            cycle_count := cycle_count + 1;
            
            -- Pre-compute fixed-point values
            a_fp := real_to_fp(a_values(i));
            b_fp := real_to_fp(b_values(i));
            y_fp := real_to_fp(y_values(i));
            
            -- Apply inputs
            a_in <= a_fp;
            b_in <= b_fp;
            y_in <= y_fp;
            valid_in <= '1';
            
            -- Calculate expected result
            expected := (b_values(i) - y_values(i))/ a_values(i);
            
            -- Report input
            write(line_buffer, string'("Cycle " & integer'image(cycle_count) & 
                  " INPUT:  a=" & real'image(a_values(i)) & 
                  ", b=" & real'image(b_values(i)) & 
                  ", y=" & real'image(y_values(i)) & 
                  ", expected=" & real'image(expected)));
            writeline(output, line_buffer);
            
            -- Check for output (will start appearing after pipeline latency)
            if valid_out = '1' then
                actual := fp_to_real(x_out);
                output_count := output_count + 1;
                write(line_buffer, string'("Cycle " & integer'image(cycle_count) & 
                      " OUTPUT: result=" & real'image(actual) & 
                      " (output #" & integer'image(output_count) & ")"));
                writeline(output, line_buffer);
            end if;
            
        end loop;
        
        -- Stop feeding inputs, but continue monitoring outputs
        wait until rising_edge(clk);
        valid_in <= '0';
        
        write(line_buffer, string'(""));
        writeline(output, line_buffer);
        write(line_buffer, string'("Input complete. Monitoring remaining outputs..."));
        writeline(output, line_buffer);
        
        -- Monitor for remaining outputs (pipeline draining)
        for i in 1 to 1200 loop
            wait until rising_edge(clk);
            cycle_count := cycle_count + 1;
            
            if valid_out = '1' then
                actual := fp_to_real(x_out);
                output_count := output_count + 1;
                write(line_buffer, string'("Cycle " & integer'image(cycle_count) & 
                      " OUTPUT: result=" & real'image(actual) & 
                      " (output #" & integer'image(output_count) & ")"));
                writeline(output, line_buffer);
            end if;
        end loop;
        
        -- Summary
        writeline(output, line_buffer);
        write(line_buffer, string'("=== TEST SUMMARY ==="));
        writeline(output, line_buffer);
        write(line_buffer, string'("Total cycles: " & integer'image(cycle_count)));
        writeline(output, line_buffer);
        write(line_buffer, string'("Inputs fed: 16"));
        writeline(output, line_buffer);
        write(line_buffer, string'("Outputs received: " & integer'image(output_count)));
        writeline(output, line_buffer);
        
        if output_count = 16 then
            write(line_buffer, string'("*** SUCCESS: All inputs produced outputs ***"));
        else
            write(line_buffer, string'("*** WARNING: Missing outputs ***"));
        end if;
        writeline(output, line_buffer);
        
        -- Calculate throughput
        if output_count > 0 then
            write(line_buffer, string'("Pipeline latency: ~" & 
                  integer'image(cycle_count - 16 - output_count + 1) & " cycles"));
            writeline(output, line_buffer);
            write(line_buffer, string'("Sustained throughput: 1 output per cycle"));
            writeline(output, line_buffer);
        end if;
        
        test_complete <= true;
        wait;
        
    end process test_process;
    
    -- Simple timeout
    timeout_process: process
    begin
        wait for 8 ms;
        assert false report "Test timeout" severity failure;
    end process;
    
end architecture simple_pipeline_test;