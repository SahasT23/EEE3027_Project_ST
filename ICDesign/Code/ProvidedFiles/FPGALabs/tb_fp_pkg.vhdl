library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use work.fp_pkg.all;

entity tb_fp_pkg is
end entity;

architecture sim of tb_fp_pkg is
  -- No DUT: we're testing pure functions in the package.
  signal done : boolean := false;

  -- Helpers for expected Q16.16 values
  function to_fp_const(r : real) return fp32 is
  begin
    return to_signed(integer(r * real(2**FP_FRAC_BITS)), 32);
  end function;

  -- Compare two fp32 with exact match (useful for integers / clean cases)
  procedure expect_eq(sig : in fp32; exp : in fp32; msg : in string) is
  begin
    assert sig = exp report "FAIL: " & msg severity error;
  end procedure;

  -- Compare real values with tolerance (for nicer console output)
  function almost_equal(a, b : real; eps : real := 1.0E-5) return boolean is
  begin
    return abs(a - b) <= eps;
  end function;

  procedure expect_real(sig : in fp32; exp_r : in real; msg : in string) is
    variable got_r : real;
  begin
    got_r := fp_to_real(sig);
    if not almost_equal(got_r, exp_r) then
      report "FAIL: " & msg & " got=" & real'image(got_r) & " exp=" & real'image(exp_r) severity error;
    end if;
  end procedure;

begin

  process
    variable L : line;
    variable a,b,res : fp32;
    -- for saturation tests
    constant MAX_POS : fp32 := to_signed(2**31 - 1, 32);
    constant MIN_NEG : fp32 := to_signed(-2**31,     32);
  begin
    -- Banner
    write(L, string'("=== Fixed-Point Package Testbench (Q16.16) ==="));
    writeline(output, L);

    --------------------------------------------------------------------
    -- Conversions
    --------------------------------------------------------------------
    a := int_to_fp(1);
    expect_eq(a, to_fp_const(1.0), "int_to_fp(1)");

    a := int_to_fp(-2);
    expect_eq(a, to_fp_const(-2.0), "int_to_fp(-2)");

    a := real_to_fp(0.5);
    expect_real(a, 0.5, "real_to_fp(0.5)");

    a := real_to_fp(-0.25);
    expect_real(a, -0.25, "real_to_fp(-0.25)");

    -- fp_to_real spot checks
    a := to_fp_const(3.125); -- 3 + 1/8
    expect_real(a, 3.125, "fp_to_real(3.125)");

    --------------------------------------------------------------------
    -- Addition
    --------------------------------------------------------------------
    a := to_fp_const(0.5);
    b := to_fp_const(0.25);
    res := fp_add(a,b);
    expect_real(res, 0.75, "0.5 + 0.25");

    a := to_fp_const(-1.0);
    b := to_fp_const(0.25);
    res := fp_add(a,b);
    expect_real(res, -0.75, "-1 + 0.25");

    --------------------------------------------------------------------
    -- Subtraction
    --------------------------------------------------------------------
    a := to_fp_const(1.0);
    b := to_fp_const(0.25);
    res := fp_sub(a,b);
    expect_real(res, 0.75, "1 - 0.25");

    a := to_fp_const(-0.5);
    b := to_fp_const(0.75);
    res := fp_sub(a,b);
    expect_real(res, -1.25, "-0.5 - 0.75");

    --------------------------------------------------------------------
    -- Multiplication
    --------------------------------------------------------------------
    a := to_fp_const(2.0);
    b := to_fp_const(0.25);
    res := fp_mul(a,b);
    expect_real(res, 0.5, "2 * 0.25");

    a := to_fp_const(-0.5);
    b := to_fp_const(0.5);
    res := fp_mul(a,b);
    expect_real(res, -0.25, "-0.5 * 0.5");

    --------------------------------------------------------------------
    -- Saturation (add/sub). This depends on your fp_add/fp_sub implementation.
    -- We'll attempt to overflow positively and negatively:
    --------------------------------------------------------------------
    a := to_signed(2**31 - 1000, 32); -- large positive close to MAX
    b := to_signed(2000, 32);
    res := fp_add(a,b);
    assert res = MAX_POS report "FAIL: positive saturation" severity error;

    a := to_signed(-2**31 + 1000, 32); -- large negative close to MIN
    b := to_signed(-2000, 32);
    res := fp_sub(a,b); -- (-2^31+1000) - (-2000) = (-2^31+3000) -> MIN_NEG when saturated if overflow detected
    assert res = MIN_NEG report "FAIL: negative saturation" severity error;

    --------------------------------------------------------------------
    -- Console summary
    --------------------------------------------------------------------
    write(L, string'("All tests completed. Check for any 'FAIL' above."));
    writeline(output, L);

    wait for 10 ns;
    done <= true;
    wait;
  end process;

end architecture;