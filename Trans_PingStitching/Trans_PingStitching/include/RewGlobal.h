//
// MATLAB Compiler: 6.5 (R2017b)
// Date: Wed Mar 25 21:01:46 2020
// Arguments:
// "-B""macro_default""-W""cpplib:RewGlobal""-T""link:lib""globalTransEsti"
//

#ifndef __RewGlobal_h
#define __RewGlobal_h 1

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
#ifndef LIB_RewGlobal_C_API 
#define LIB_RewGlobal_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_RewGlobal_C_API 
bool MW_CALL_CONV RewGlobalInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_RewGlobal_C_API 
bool MW_CALL_CONV RewGlobalInitialize(void);

extern LIB_RewGlobal_C_API 
void MW_CALL_CONV RewGlobalTerminate(void);

extern LIB_RewGlobal_C_API 
void MW_CALL_CONV RewGlobalPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_RewGlobal_C_API 
bool MW_CALL_CONV mlxGlobalTransEsti(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_RewGlobal
#define PUBLIC_RewGlobal_CPP_API __declspec(dllexport)
#else
#define PUBLIC_RewGlobal_CPP_API __declspec(dllimport)
#endif

#define LIB_RewGlobal_CPP_API PUBLIC_RewGlobal_CPP_API

#else

#if !defined(LIB_RewGlobal_CPP_API)
#if defined(LIB_RewGlobal_C_API)
#define LIB_RewGlobal_CPP_API LIB_RewGlobal_C_API
#else
#define LIB_RewGlobal_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_RewGlobal_CPP_API void MW_CALL_CONV globalTransEsti(int nargout, mwArray& paras, mwArray& R_ref, const mwArray& X, const mwArray& imsize, const mwArray& edge_list);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
