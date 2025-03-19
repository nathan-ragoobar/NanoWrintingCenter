#pragma once

// Prevent clock_gettime redefinition issues
#ifdef __MINGW32__
  #ifndef MINGW_HAS_SECURE_API
    #define MINGW_HAS_SECURE_API 1
  #endif
  
  #ifdef clock_gettime
    #undef clock_gettime
  #endif

  #ifdef CLOCK_MONOTONIC
    #undef CLOCK_MONOTONIC
  #endif
#endif

// Fix include paths for Eigen ThreadPool
#include <Eigen/Core>
#ifdef EIGEN_USE_THREADS
  #include <unsupported/Eigen/CXX11/ThreadPool>
#endif