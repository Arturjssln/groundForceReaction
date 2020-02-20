#ifndef _OSIM_PLUGIN_DLL_H_
#define _OSIM_PLUGIN_DLL_H_

#ifndef WIN32
#define OSIMPLUGIN_API
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#ifdef OSIMPLUGIN_EXPORTS
#define OSIMPLUGIN_API __declspec(dllexport)
#else
#define OSIMPLUGIN_API __declspec(dllimport)
#endif
#endif

#endif