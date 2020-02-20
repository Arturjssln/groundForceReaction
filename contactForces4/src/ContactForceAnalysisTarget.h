/*************************************************************************
*                            <contactForces>
*                           Copyright (c) 2016
*                Engineering Design (KTmfk) and Daniel Krüger
*                      Universität Erlangen-Nürnberg
*                            All rights reserved
*
* <contactForces> is a plugin for OpenSim to estimate unknown external forces
* within inverse dynamic simulations.
* The plugin provides special contact force components and an associated analysis procedure
* that solves an extended static optimization problem taking into account friction constraints.
* <contactForces> is developed at the Engineering Desing institute (KTmfk),
* Universität Erlangen-Nürnberg.
*
* Copyright (c) 2016 Engineering Design (KTmfk) and Daniel Krüger
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*           http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* This Software uses the OpenSim API a toolkit for musculoskeletal modeling
* and simulation. See http://opensim.stanford.edu for mor information
************************************************************************
* Author: Daniel Krüger			daniel.krueger@fau.de
* Co-Dev: Felix Laufer			laufer@cs.uni-kl.de
*******************************************************************/

#ifndef _OSIM_CONTACT_FORCE_ANALYSIS_TARGET_H_
#define _OSIM_CONTACT_FORCE_ANALYSIS_TARGET_H_

#include "ContactForceAnalysis.h"

#include <OpenSim/OpenSim.h>
#include <vector>

namespace OpenSim
{
  class OSIMPLUGIN_API ContactForceAnalysisTarget : public SimTK::OptimizerSystem
  {
  public:  
    explicit ContactForceAnalysisTarget(OpenSim::ContactForceAnalysis* analysis);
    ~ContactForceAnalysisTarget();

    void initialize(const SimTK::State& s);
    SimTK::Vector initParameters() const;
    void setForces(const SimTK::Vector& parameters, SimTK::State& s);
    SimTK::String printPerformance(const SimTK::Vector& parameters) const;

    int constraintFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Vector& errors) const override;
    int constraintJacobian(const SimTK::Vector& parameters, bool new_parameters, SimTK::Matrix& jacobian) const override;
    int objectiveFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Real& objective) const override;
    int gradientFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Vector& gradient) const override;

  private:
    void computeConstraints(const SimTK::Vector& parameters, const SimTK::Vector& constraints);

    ContactForceAnalysis* _analysis;
    SimTK::State _si;
    std::vector<int> _qs2FreeQs;
    int _numResiduals;
    double _systemMass;
    SimTK::Vector _optimalForces;
    mutable SimTK::Matrix _constraintMatrix;
    mutable SimTK::Vector _constraintVector;
  };
}

#endif