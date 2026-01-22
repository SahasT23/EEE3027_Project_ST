--------------------------------------------------------------------------------
-- tb_fp_pkg.vhdl - Testbench for Fixed-Point Arithmetic Package
-- EEE3027 IC Design Labs
--
-- Tests all conversion and arithmetic functions in fp_pkg
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.fp_pkg.all;

entity tb_fp_pkg is
end;

architecture sim of tb_fp_pkg is
  signal test_a, test_b, result : fp32;
begin

  process
    variable a, b, r : fp32;
    variable real_val : real;
  begin
    report "=== Testing fp_pkg Functions ===" severity note;
    
    --------------------------------------------------------------------------
    -- Test Conversions
    --------------------------------------------------------------------------
    report "--- Conversion Tests ---" severity note;
    
    -- int_to_fp
    a := int_to_fp(5);
    report "int_to_fp(5) = " & real'image(fp_to_real(a)) & " (expected 5.0)";
    
    a := int_to_fp(-3);
    report "int_to_fp(-3) = " & real'image(fp_to_real(a)) & " (expected -3.0)";
    
    -- real_to_fp
    a := real_to_fp(3.14159);
    report "real_to_fp(3.14159) = " & real'image(fp_to_real(a));
    
    a := real_to_fp(-2.5);
    report "real_to_fp(-2.5) = " & real'image(fp_to_real(a)) & " (expected -2.5)";
    
    --------------------------------------------------------------------------
    -- Test Addition
    --------------------------------------------------------------------------
    report "--- Addition Tests ---" severity note;
    
    -- Simple add
    a := real_to_fp(1.5);
    b := real_to_fp(2.25);
    r := fp_add(a, b);
    report "1.5 + 2.25 = " & real'image(fp_to_real(r)) & " (expected 3.75)";
    
    -- Add negative
    a := real_to_fp(5.0);
    b := real_to_fp(-3.0);
    r := fp_add(a, b);
    report "5.0 + (-3.0) = " & real'image(fp_to_real(r)) & " (expected 2.0)";
    
    -- Positive overflow saturation
    a := FP_MAX;
    b := FP_ONE;
    r := fp_add(a, b);
    report "MAX + 1.0 = " & real'image(fp_to_real(r)) & " (should saturate to MAX)";
    
    --------------------------------------------------------------------------
    -- Test Subtraction
    --------------------------------------------------------------------------
    report "--- Subtraction Tests ---" severity note;
    
    a := real_to_fp(5.0);
    b := real_to_fp(2.0);
    r := fp_sub(a, b);
    report "5.0 - 2.0 = " & real'image(fp_to_real(r)) & " (expected 3.0)";
    
    a := real_to_fp(1.0);
    b := real_to_fp(4.0);
    r := fp_sub(a, b);
    report "1.0 - 4.0 = " & real'image(fp_to_real(r)) & " (expected -3.0)";
    
    -- Negative overflow saturation
    a := FP_MIN;
    b := FP_ONE;
    r := fp_sub(a, b);
    report "MIN - 1.0 = " & real'image(fp_to_real(r)) & " (should saturate to MIN)";
    
    --------------------------------------------------------------------------
    -- Test Multiplication
    --------------------------------------------------------------------------
    report "--- Multiplication Tests ---" severity note;
    
    a := real_to_fp(2.0);
    b := real_to_fp(3.0);
    r := fp_mul(a, b);
    report "2.0 * 3.0 = " & real'image(fp_to_real(r)) & " (expected 6.0)";
    
    a := real_to_fp(0.5);
    b := real_to_fp(0.5);
    r := fp_mul(a, b);
    report "0.5 * 0.5 = " & real'image(fp_to_real(r)) & " (expected 0.25)";
    
    a := real_to_fp(-2.0);
    b := real_to_fp(3.0);
    r := fp_mul(a, b);
    report "-2.0 * 3.0 = " & real'image(fp_to_real(r)) & " (expected -6.0)";
    
    a := real_to_fp(-2.5);
    b := real_to_fp(-4.0);
    r := fp_mul(a, b);
    report "-2.5 * -4.0 = " & real'image(fp_to_real(r)) & " (expected 10.0)";
    
    --------------------------------------------------------------------------
    -- Test Combined Operations (MAC)
    --------------------------------------------------------------------------
    report "--- MAC Operation Test ---" severity note;
    
    a := real_to_fp(2.0);
    b := real_to_fp(3.0);
    r := fp_mul(a, b);        -- 2.0 * 3.0 = 6.0
    r := fp_add(r, real_to_fp(1.5));  -- 6.0 + 1.5 = 7.5
    report "2.0 * 3.0 + 1.5 = " & real'image(fp_to_real(r)) & " (expected 7.5)";
    
    report "=== fp_pkg Tests Complete ===" severity note;
    wait;
  end process;

end;
