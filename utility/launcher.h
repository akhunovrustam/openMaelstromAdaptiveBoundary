#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
//#include <nvfunctional>
#include <regex>
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#ifdef __CUDACC__
#pragma nv_diag_suppress = integer_sign_change
#endif
#include <tbb/blocked_range.h> 
#include <tbb/parallel_for.h> 
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __CUDACC__
#pragma nv_diag_default = integer_sign_change
#endif
#include <utility/cuda/cache.h>
#include <utility/cuda/error_handling.h>
#include <tools/color.h>
#include <tools/log.h>
#include <tools/timer.h>
#include <utility/cuda.h>
#include <utility/launcher/compute_launcher.h>
#include <utility/launcher/launch_functions.h>
#include <utility/launcher/macro.h>