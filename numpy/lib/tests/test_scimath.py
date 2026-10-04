import pytest

import numpy as np
from numpy.testing import assert_equal

# The complex dtype that the emath functions return for inputs outside the
# real domain: the complex type matching the precision of the real result
# (complex64 for float16, which has no complex counterpart).
COMPLEX_DTYPES = [
    (np.int8, np.complex64),
    (np.uint8, np.complex64),
    (np.int16, np.complex64),
    (np.uint16, np.complex64),
    (np.int32, np.complex128),
    (np.uint32, np.complex128),
    (np.int64, np.complex128),
    (np.uint64, np.complex128),
    (np.float16, np.complex64),
    (np.float32, np.complex64),
    (np.float64, np.complex128),
    (np.longdouble, np.clongdouble),
    (np.complex64, np.complex64),
    (np.complex128, np.complex128),
    (np.clongdouble, np.clongdouble),
]
# Unsigned integers cannot be negative.
SIGNED_COMPLEX_DTYPES = [
    (dtype, cdtype) for dtype, cdtype in COMPLEX_DTYPES
    if not np.issubdtype(dtype, np.unsignedinteger)
]
DTYPES = [dtype for dtype, _ in COMPLEX_DTYPES]


def inputs(values, dtype):
    # Use the full precision of inexact dtypes, so that computing in a lower
    # precision changes the results.
    x = np.array(values, dtype=dtype)
    if np.issubdtype(dtype, np.inexact):
        x *= 1 + np.finfo(dtype).eps
    return x


class TestComplexResult:
    @pytest.mark.parametrize("func", ["sqrt", "log", "log2", "log10"])
    @pytest.mark.parametrize(("dtype", "cdtype"), SIGNED_COMPLEX_DTYPES)
    def test_negative_input(self, func, dtype, cdtype):
        x = inputs([-4, 4], dtype)
        res = getattr(np.emath, func)(x)
        assert res.dtype == cdtype
        assert_equal(res, getattr(np, func)(x.astype(cdtype)))

    @pytest.mark.parametrize("func", ["arccos", "arcsin", "arctanh"])
    @pytest.mark.parametrize(("dtype", "cdtype"), COMPLEX_DTYPES)
    def test_input_beyond_one(self, func, dtype, cdtype):
        x = inputs([3, 0], dtype)
        res = getattr(np.emath, func)(x)
        assert res.dtype == cdtype
        assert_equal(res, getattr(np, func)(x.astype(cdtype)))

    @pytest.mark.parametrize(("dtype", "cdtype"), SIGNED_COMPLEX_DTYPES)
    def test_logn(self, dtype, cdtype):
        n = inputs([2, -2], dtype)
        x = inputs([-4, -4], dtype)
        res = np.emath.logn(n, x)
        assert res.dtype == cdtype
        assert_equal(res, np.log(x.astype(cdtype)) / np.log(n.astype(cdtype)))
        # Only x outside the real domain.
        res = np.emath.logn(n[:1], x)
        assert res.dtype == cdtype
        assert_equal(res, np.log(x.astype(cdtype)) / np.log(n[:1]))

    @pytest.mark.parametrize(("dtype", "cdtype"), SIGNED_COMPLEX_DTYPES)
    def test_power(self, dtype, cdtype):
        x = inputs([-4, 4], dtype)
        p = np.array([2, 3], dtype=dtype)
        res = np.emath.power(x, p)
        assert res.dtype == cdtype
        assert_equal(res, np.power(x.astype(cdtype), p))

    @pytest.mark.parametrize(("dtype", "cdtype"), [
        (np.float16, np.complex64),
        (np.longdouble, np.clongdouble),
        (np.clongdouble, np.clongdouble),
    ])
    def test_scalar(self, dtype, cdtype):
        assert type(np.emath.sqrt(dtype(-4))) is cdtype
        assert type(np.emath.arcsin(dtype(3))) is cdtype

    def test_longdouble_precision(self):
        # gh-28367: long doubles used to be computed in double precision,
        # where -(1 + eps) rounds to -1 (unless long double is double).
        x = -(1 + np.finfo(np.longdouble).eps)
        assert np.emath.log(x).real > 0


class TestRealResult:
    @pytest.mark.parametrize(
        "func", ["sqrt", "log", "log2", "log10", "arccos", "arcsin", "arctanh"])
    @pytest.mark.parametrize("dtype", [np.bool, *DTYPES])
    def test_input_in_domain(self, func, dtype):
        # Inputs in the domain give the same result as the ufunc.
        if func.startswith("arc"):
            x = np.array([0, 0], dtype=dtype)
        else:
            x = np.array([1, 1], dtype=dtype)
        res = getattr(np.emath, func)(x)
        expected = getattr(np, func)(x)
        assert res.dtype == expected.dtype
        assert_equal(res, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_logn_and_power_in_domain(self, dtype):
        n = np.array([2, 4], dtype=dtype)
        x = np.array([4, 2], dtype=dtype)
        res = np.emath.logn(n, x)
        expected = np.log(x) / np.log(n)
        assert res.dtype == expected.dtype
        assert_equal(res, expected)
        res = np.emath.power(x, n)
        expected = np.power(x, n)
        assert res.dtype == expected.dtype
        assert_equal(res, expected)


class TestPowerPromotion:
    # Like in np.power, Python scalars are weakly typed (NEP 50), so that,
    # e.g., float32 ** 0.5 is float32 (and complex64 for negative bases).

    @pytest.mark.parametrize("p", [
        2, 0.5, -2, -0.5, 2j, np.float32(0.5), np.int16(2),
        np.array([2, 0.5], dtype=np.float32),
    ], ids=repr)
    @pytest.mark.parametrize("x", [
        *(np.array([4, 9], dtype=dtype) for dtype in [
            np.int8, np.uint8, np.int64, np.float16, np.float32, np.float64,
            np.longdouble, np.complex64]),
        np.int8(4), np.float32(4), 4, 4.0, 4 + 0j,
    ], ids=repr)
    def test_like_power(self, x, p):
        res = np.emath.power(x, p)
        # Integers cannot be raised to negative integer powers.
        expected = np.power(x, float(p) if type(p) is int and p < 0 else p)
        assert type(res) is type(expected)
        assert res.dtype == expected.dtype
        assert_equal(res, expected)

    @pytest.mark.parametrize(("x", "p", "cdtype"), [
        (np.array([-4, 4], dtype=np.float32), 0.5, np.complex64),
        (np.array([-4, 4], dtype=np.float32), 2, np.complex64),
        (np.array([-4, 4], dtype=np.float32), -2, np.complex64),
        (np.array([-4, 4], dtype=np.float32), np.float64(0.5), np.complex128),
        (np.array([-4, 4], dtype=np.float16), 0.5, np.complex64),
        (np.array([-4, 4], dtype=np.float64), 0.5, np.complex128),
        (np.array([-4, 4], dtype=np.longdouble), 0.5, np.clongdouble),
        (np.array([-4, 4], dtype=np.complex64), 0.5, np.complex64),
        (np.array([-4, 4], dtype=np.int8), 2, np.complex64),
        # Like np.power(int8, 0.5), which is float64.
        (np.array([-4, 4], dtype=np.int8), 0.5, np.complex128),
        (np.array([-4, 4], dtype=np.int64), 0.5, np.complex128),
        (np.float32(-4), 0.5, np.complex64),
        (-4, np.float32(0.5), np.complex64),
        (-4.0, np.array([0.5, 2], dtype=np.float32), np.complex64),
        (-4, 0.5, np.complex128),
        (-4, 2, np.complex128),
    ], ids=repr)
    def test_negative_base(self, x, p, cdtype):
        res = np.emath.power(x, p)
        assert res.dtype == cdtype
        assert_equal(res, np.power(np.asarray(x).astype(cdtype), p))
        if np.ndim(x) == np.ndim(p) == 0:
            assert type(res) is cdtype

    @pytest.mark.parametrize(("x", "p", "expected"), [
        (np.array([2, 4]), np.array([-1, 2]), np.array([0.5, 16])),
        (np.array([2, 4], dtype=np.int8), -1, np.array([0.5, 0.25])),
        (2, np.int8(-1), np.float64(0.5)),
    ], ids=repr)
    def test_negative_integer_exponent(self, x, p, expected):
        # Integers cannot be raised to negative integer powers, so these
        # exponents are made float.
        res = np.emath.power(x, p)
        assert type(res) is type(expected)
        assert res.dtype == expected.dtype
        assert_equal(res, expected)
