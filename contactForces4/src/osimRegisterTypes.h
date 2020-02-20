#ifndef _OSIM_REGISTER_TYPES_H_
#define _OSIM_REGISTER_TYPES_H_

#include "osimPluginDLL.h"

extern "C"
{
  OSIMPLUGIN_API void RegisterTypes();
}

class dllObjectInstantiator
{
public:
  dllObjectInstantiator()
  {
    registerDllClasses();
  }

private:
  static void registerDllClasses()
  {
    RegisterTypes();
  }
};

#endif