--------------------------------------------------------------------------------
-- fp_pkg.vhdl - Fixed-Point Arithmetic Package (Q16.16 Format)
-- EEE3027 IC Design Labs
-- 
-- This package provides Q16.16 fixed-point arithmetic operations:
--   - 16 integer bits (including sign)
--   - 16 fractional bits
--   - Range: approximately -32768 to +32767.99998
--   - Resolution: 2^-16 ≈ 0.0000153
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package fp_pkg is
  -- Q16.16 fixed-point type: 32-bit signed
  subtype fp32 is signed(31 downto 0);

  -- Number of fractional bits
  constant FP_FRAC_BITS : integer := 16;

  -- Saturation constants for overflow handling
  constant FP_MAX : fp32 := to_signed(2147483647, 32);   -- Max positive: 0x7FFFFFFF
  constant FP_MIN : fp32 := to_signed(-2147483648, 32);  -- Max negative: 0x80000000

  -- Common constants
  constant FP_ZERO : fp32 := (others => '0');
  constant FP_ONE  : fp32 := to_signed(2**FP_FRAC_BITS, 32);  -- 1.0 = 0x00010000

  -- === Conversion Functions (simulation-only, non-synthesizable) ===
  function int_to_fp(i : integer) return fp32;
  function real_to_fp(r : real) return fp32;
  function fp_to_real(x : fp32) return real;

  -- === Arithmetic Functions (synthesizable) ===
  function fp_add(a, b : fp32) return fp32;  -- Addition with saturation
  function fp_sub(a, b : fp32) return fp32;  -- Subtraction with saturation
  function fp_mul(a, b : fp32) return fp32;  -- Q16.16 multiplication

end package;

package body fp_pkg is

  ----------------------------------------------------------------------------
  -- int_to_fp: Convert integer to Q16.16 fixed-point
  -- Left-shifts by 16 bits to place integer in upper portion
  ----------------------------------------------------------------------------
  function int_to_fp(i : integer) return fp32 is
    variable tmp : integer;
  begin
    tmp := i * (2**FP_FRAC_BITS);  -- Shift left by 16 bits
    return to_signed(tmp, 32);
  end function;

  ----------------------------------------------------------------------------
  -- real_to_fp: Convert real to Q16.16 (simulation-only)
  -- Multiplies by 2^16 and truncates to integer
  ----------------------------------------------------------------------------
  function real_to_fp(r : real) return fp32 is
    variable scaled : integer;
  begin
    scaled := integer(r * real(2**FP_FRAC_BITS));
    return to_signed(scaled, 32);
  end function;

  ----------------------------------------------------------------------------
  -- fp_to_real: Convert Q16.16 to real (simulation-only)
  -- Divides by 2^16 to recover real value
  ----------------------------------------------------------------------------
  function fp_to_real(x : fp32) return real is
  begin
    return real(to_integer(x)) / real(2**FP_FRAC_BITS);
  end function;

  ----------------------------------------------------------------------------
  -- fp_add: Fixed-point addition with saturation
  -- Extends to 33 bits to detect overflow, saturates if needed
  ----------------------------------------------------------------------------
  function fp_add(a, b : fp32) return fp32 is
    variable ext_a, ext_b, sum : signed(32 downto 0);  -- 33-bit for overflow detection
  begin
    -- Sign-extend both operands to 33 bits
    ext_a := signed'(a(31) & a);
    ext_b := signed'(b(31) & b);
    sum   := ext_a + ext_b;
    
    -- Check for overflow: MSB differs from sign bit
    if (sum(32) /= sum(31)) then
      if sum(32) = '0' then
        return FP_MAX;  -- Positive overflow -> saturate to max
      else
        return FP_MIN;  -- Negative overflow -> saturate to min
      end if;
    else
      return sum(31 downto 0);  -- No overflow, return result
    end if;
  end function;

  ----------------------------------------------------------------------------
  -- fp_sub: Fixed-point subtraction with saturation
  -- Same overflow detection as addition
  ----------------------------------------------------------------------------
  function fp_sub(a, b : fp32) return fp32 is
    variable ext_a, ext_b, diff : signed(32 downto 0);
  begin
    ext_a := signed'(a(31) & a);
    ext_b := signed'(b(31) & b);
    diff  := ext_a - ext_b;
    
    -- Check for overflow
    if (diff(32) /= diff(31)) then
      if diff(32) = '0' then
        return FP_MAX;
      else
        return FP_MIN;
      end if;
    else
      return diff(31 downto 0);
    end if;
  end function;

  ----------------------------------------------------------------------------
  -- fp_mul: Fixed-point multiplication
  -- Q16.16 * Q16.16 = Q32.32 -> extract bits [47:16] for Q16.16 result
  -- Maps to DSP48E1 blocks on Artix-7 FPGA
  ----------------------------------------------------------------------------
  function fp_mul(a, b : fp32) return fp32 is
    variable prod64 : signed(63 downto 0);  -- 64-bit product
    variable a32    : signed(31 downto 0);
    variable b32    : signed(31 downto 0);
    variable res    : fp32;
  begin
    a32    := a;
    b32    := b;
    prod64 := a32 * b32;           -- 32x32 -> 64-bit signed multiply
    res    := prod64(47 downto 16); -- Extract middle 32 bits (Q16.16)
    return res;
  end function;

end package body;
