library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
-- Note: real-based conversions are for simulation only (non-synthesizable).

package fp_pkg is
  -- Q16.16 fixed-point type
  subtype fp32 is signed(31 downto 0);

  constant FP_FRAC_BITS : integer := 16;

  constant FP_MAX  : fp32 := to_signed(2147483647, 32);
  constant FP_MIN  : fp32 := to_signed(-2147483648, 32);


  -- Common constants
  constant FP_ZERO : fp32 := (others => '0');
  constant FP_ONE  : fp32 := to_signed(2**FP_FRAC_BITS, 32);

  -- === Conversions (simulation-friendly) ===
  function int_to_fp(i : integer) return fp32;
  function real_to_fp(r : real) return fp32;   -- simulation-only
  function fp_to_real(x : fp32) return real;   -- simulation-only

  -- === Arithmetic (synthesizable) ===
  function fp_add(a, b : fp32) return fp32;        -- with saturation
  function fp_sub(a, b : fp32) return fp32;        -- with saturation
  function fp_mul(a, b : fp32) return fp32;        -- Q16.16 * Q16.16 -> Q16.16

end package;

package body fp_pkg is

  function int_to_fp(i : integer) return fp32 is
    variable tmp : integer;
  begin
    tmp := i * (2**FP_FRAC_BITS);
    return to_signed(tmp, 32);
  end function;

  function real_to_fp(r : real) return fp32 is
    variable scaled : integer;
  begin
    -- Truncate (avoid 'round' to keep dependencies minimal)
    scaled := integer(r * real(2**FP_FRAC_BITS));
    return to_signed(scaled, 32);
  end function;

  function fp_to_real(x : fp32) return real is
  begin
    return real(to_integer(x)) / real(2**FP_FRAC_BITS);
  end function;

  function fp_add(a, b : fp32) return fp32 is
    variable ext_a, ext_b, sum : signed(32 downto 0);
  begin
    ext_a := signed'(a(31) & a);
    ext_b := signed'(b(31) & b);
    sum   := ext_a + ext_b;
    -- Saturate on overflow
    if (sum(32) /= sum(31)) then
      if sum(32) = '0' then
        return FP_MAX; -- positive overflow
      else
        return FP_MIN; -- negative overflow
      end if;
    else
      return sum(31 downto 0);
    end if;
  end function;

  function fp_sub(a, b : fp32) return fp32 is
    variable ext_a, ext_b, diff : signed(32 downto 0);
  begin
    ext_a := signed'(a(31) & a);
    ext_b := signed'(b(31) & b);
    diff  := ext_a - ext_b;
    -- Saturate on overflow
    if (diff(32) /= diff(31)) then
      if diff(32) = '0' then
        return FP_MAX; -- positive overflow
      else
        return FP_MIN; -- negative overflow
      end if;
    else
      return diff(31 downto 0);
    end if;
  end function;

  function fp_mul(a, b : fp32) return fp32 is
    -- 32x32 signed multiply -> 64-bit product
    variable prod64 : signed(63 downto 0);
    variable a32    : signed(31 downto 0);
    variable b32    : signed(31 downto 0);
    variable res    : fp32;
  begin
    a32    := a;
    b32    := b;
    prod64 := a32 * b32;         -- numeric_std returns 64b here
    -- Q16.16 * Q16.16 = Q32.32 -> take bits [47:16]
    res := prod64(47 downto 16);
    return res;
  end function;

end package body;