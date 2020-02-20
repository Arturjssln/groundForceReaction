#include "osimRegisterTypes.h"
#include "ContactPointOnPlane.h"
#include "ContactForceAnalysis.h"

#include <OpenSim/Common/Object.h>

static dllObjectInstantiator instantiator;

OSIMPLUGIN_API void RegisterTypes()
{
  OpenSim::Object::RegisterType(OpenSim::ContactForceAnalysis());
  OpenSim::Object::RegisterType(OpenSim::ContactPointOnPlane());
}