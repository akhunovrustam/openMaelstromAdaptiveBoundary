#pragma once
#include <cmath>

/* single precision constants */
#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#define CUDART_MIN_DENORM_F     __int_as_float(0x00000001)
#define CUDART_MAX_NORMAL_F     __int_as_float(0x7f7fffff)
#define CUDART_NEG_ZERO_F       __int_as_float(0x80000000)
#define CUDART_ZERO_F           0.0f
#define CUDART_ONE_F            1.0f
#define CUDART_SQRT_HALF_F      0.707106781f
#define CUDART_SQRT_HALF_HI_F   0.707106781f
#define CUDART_SQRT_HALF_LO_F   1.210161749e-08f
#define CUDART_SQRT_TWO_F       1.414213562f
#define CUDART_THIRD_F          0.333333333f
#define CUDART_PIO4_F           0.785398163f
#define CUDART_PIO2_F           1.570796327f
#define CUDART_3PIO4_F          2.356194490f
#define CUDART_2_OVER_PI_F      0.636619772f
#define CUDART_SQRT_2_OVER_PI_F 0.797884561f
#define CUDART_PI_F             3.141592654f
#define CUDART_L2E_F            1.442695041f
#define CUDART_L2T_F            3.321928094f
#define CUDART_LG2_F            0.301029996f
#define CUDART_LGE_F            0.434294482f
#define CUDART_LN2_F            0.693147181f
#define CUDART_LNT_F            2.302585093f 
#define CUDART_LNPI_F           1.144729886f
#define CUDART_TWO_TO_M126_F    1.175494351e-38f
#define CUDART_TWO_TO_126_F     8.507059173e37f
#define CUDART_NORM_HUGE_F      3.402823466e38f
#define CUDART_TWO_TO_23_F      8388608.0f
#define CUDART_TWO_TO_24_F      16777216.0f
#define CUDART_TWO_TO_31_F      2147483648.0f
#define CUDART_TWO_TO_32_F      4294967296.0f
#define CUDART_REMQUO_BITS_F    3
#define CUDART_REMQUO_MASK_F    (~((~0)<<CUDART_REMQUO_BITS_F))
#define CUDART_TRIG_PLOSS_F     105615.0f

/* double precision constants */
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_NAN              __longlong_as_double(0xfff8000000000000ULL)
#define CUDART_NEG_ZERO         __longlong_as_double(0x8000000000000000ULL)
#define CUDART_MIN_DENORM       __longlong_as_double(0x0000000000000001ULL)
#define CUDART_ZERO             0.0
#define CUDART_ONE              1.0
#define CUDART_SQRT_TWO         1.4142135623730951e+0
#define CUDART_SQRT_HALF        7.0710678118654757e-1
#define CUDART_SQRT_HALF_HI     7.0710678118654757e-1
#define CUDART_SQRT_HALF_LO   (-4.8336466567264567e-17)
#define CUDART_THIRD            3.3333333333333333e-1
#define CUDART_TWOTHIRD         6.6666666666666667e-1
#define CUDART_PIO4             7.8539816339744828e-1
#define CUDART_PIO4_HI          7.8539816339744828e-1
#define CUDART_PIO4_LO          3.0616169978683830e-17
#define CUDART_PIO2             1.5707963267948966e+0
#define CUDART_PIO2_HI          1.5707963267948966e+0
#define CUDART_PIO2_LO          6.1232339957367660e-17
#define CUDART_3PIO4            2.3561944901923448e+0
#define CUDART_2_OVER_PI        6.3661977236758138e-1
#define CUDART_PI               3.1415926535897931e+0
#define CUDART_PI_HI            3.1415926535897931e+0
#define CUDART_PI_LO            1.2246467991473532e-16
#define CUDART_SQRT_2PI         2.5066282746310007e+0
#define CUDART_SQRT_2PI_HI      2.5066282746310007e+0
#define CUDART_SQRT_2PI_LO    (-1.8328579980459167e-16)
#define CUDART_SQRT_PIO2        1.2533141373155003e+0
#define CUDART_SQRT_PIO2_HI     1.2533141373155003e+0
#define CUDART_SQRT_PIO2_LO   (-9.1642899902295834e-17)
#define CUDART_SQRT_2OPI        7.9788456080286536e-1
#define CUDART_L2E              1.4426950408889634e+0
#define CUDART_L2E_HI           1.4426950408889634e+0
#define CUDART_L2E_LO           2.0355273740931033e-17
#define CUDART_L2T              3.3219280948873622e+0
#define CUDART_LG2              3.0102999566398120e-1
#define CUDART_LG2_HI           3.0102999566398120e-1
#define CUDART_LG2_LO         (-2.8037281277851704e-18)
#define CUDART_LGE              4.3429448190325182e-1
#define CUDART_LGE_HI           4.3429448190325182e-1
#define CUDART_LGE_LO           1.09831965021676510e-17
#define CUDART_LN2              6.9314718055994529e-1
#define CUDART_LN2_HI           6.9314718055994529e-1
#define CUDART_LN2_LO           2.3190468138462996e-17
#define CUDART_LNT              2.3025850929940459e+0
#define CUDART_LNT_HI           2.3025850929940459e+0
#define CUDART_LNT_LO         (-2.1707562233822494e-16)
#define CUDART_LNPI             1.1447298858494002e+0
#define CUDART_LN2_X_1024       7.0978271289338397e+2
#define CUDART_LN2_X_1025       7.1047586007394398e+2
#define CUDART_LN2_X_1075       7.4513321910194122e+2
#define CUDART_LG2_X_1024       3.0825471555991675e+2
#define CUDART_LG2_X_1075       3.2360724533877976e+2
#define CUDART_TWO_TO_23        8388608.0
#define CUDART_TWO_TO_52        4503599627370496.0
#define CUDART_TWO_TO_53        9007199254740992.0
#define CUDART_TWO_TO_54        18014398509481984.0
#define CUDART_TWO_TO_M54       5.5511151231257827e-17
#define CUDART_TWO_TO_M1022     2.22507385850720140e-308
#define CUDART_TRIG_PLOSS       2147483648.0
#define CUDART_DBL2INT_CVT      6755399441055744.0

/* bounding values */
#define MATH_INF_F HUGE_VALF
#define MATH_INF HUGE_VAL
#define MATH_NAN_F NAN

/* simple values */
#define MATH_ZERO_F           0.0f
#define MATH_ONE_F            1.0f
#define MATH_ZERO             0.0
#define MATH_ONE              1.0
#define MATH_THIRD_F          0.333333333f
#define MATH_THIRD            3.3333333333333333e-1
#define MATH_TWOTHIRD_F       0.666666666f
#define MATH_TWOTHIRD         6.6666666666666667e-1

/* sqrt defines*/
#define MATH_SQRT_HALF_F      0.707106781f
#define MATH_SQRT_HALF        7.0710678118654757e-1
#define MATH_SQRT_TWO_F       1.414213562f
#define MATH_SQRT_TWO         1.4142135623730951e+0

/* PI defines*/
#define MATH_PI_F             3.141592654f
#define MATH_PI               3.1415926535897931e+0
#define MATH_PIO4_F           0.785398163f
#define MATH_PIO4             7.8539816339744828e-1
#define MATH_PIO2_F           1.570796327f
#define MATH_PIO2             1.5707963267948966e+0
#define MATH_3PIO4_F          2.356194490f
#define MATH_3PIO4            2.3561944901923448e+0
#define MATH_4PIO3_F		  4.188790204f
#define MATH_4PIO3			  4.18879020478639098+0
#define MATH_4PIO3_1_F		  0.238732414f
#define MATH_4PIO3_1		  2.38732414637843003-1
#define MATH_2_OVER_PI_F      0.636619772f
#define MATH_2_OVER_PI        6.3661977236758138e-1
#define MATH_SQRT_2_OVER_PI_F 0.797884561f
#define MATH_SQRT_2_OVER_PI   7.9788456080286536e-1
#define MATH_SQRT_2PI_F       2.506628274f
#define MATH_SQRT_2PI         2.5066282746310007e+0
#define MATH_SQRT_PIO2_F      1.253314137f
#define MATH_SQRT_PIO2        1.2533141373155003e+0

/* LOG defines */
#define MATH_L2E_F            1.442695041f // log_2(e)
#define MATH_L2E              1.4426950408889634e+0
#define MATH_L2T_F            3.321928094f // log_2(10)
#define MATH_L2T              3.3219280948873622e+0
#define MATH_LG2_F            0.301029996f // log_10(2)
#define MATH_LG2              3.0102999566398120e-1
#define MATH_LGE_F            0.434294482f // log_10(e)
#define MATH_LGE              4.3429448190325182e-1
#define MATH_LN2_F            0.693147181f // ln(2)
#define MATH_LN2              6.9314718055994529e-1
#define MATH_LNT_F            2.302585093f // ln(10)
#define MATH_LNT              2.3025850929940459e+0
#define MATH_LNPI_F           1.144729886f // ln(pi)
#define MATH_LNPI             1.1447298858494002e+0