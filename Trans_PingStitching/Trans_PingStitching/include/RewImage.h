//
// MATLAB Compiler: 6.5 (R2017b)
// Date: Fri Apr 10 18:12:51 2020
// Arguments: "-B""macro_default""-W""cpplib:RewImage""-T""link:lib""REW"
//

#ifndef __RewImage_h
#define __RewImage_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_RewImage_C_API 
#define LIB_RewImage_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_RewImage_C_API 
bool MW_CALL_CONV RewImageInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_RewImage_C_API 
bool MW_CALL_CONV RewImageInitialize(void);

extern LIB_RewImage_C_API 
void MW_CALL_CONV RewImageTerminate(void);

extern LIB_RewImage_C_API 
void MW_CALL_CONV RewImagePrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_RewImage_C_API 
bool MW_CALL_CONV mlxREW(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_RewImage
#define PUBLIC_RewImage_CPP_API __declspec(dllexport)
#else
#define PUBLIC_RewImage_CPP_API __declspec(dllimport)
#endif

#define LIB_RewImage_CPP_API PUBLIC_RewImage_CPP_API

#else

#if !defined(LIB_RewImage_CPP_API)
#if defined(LIB_RewImage_C_API)
#define LIB_RewImage_CPP_API LIB_RewImage_C_API
#else
#define LIB_RewImage_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_RewImage_CPP_API void MW_CALL_CONV REW(int nargout, mwArray& m_u0_, mwArray& m_u1_, const mwArray& X, const mwArray& im_n);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
