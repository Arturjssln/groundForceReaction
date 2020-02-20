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

#ifndef _OSIM_CONTACT_FORCE_ANALYSIS_H_
#define _OSIM_CONTACT_FORCE_ANALYSIS_H_

#include "osimPluginDLL.h"
#include "ContactPointOnPlane.h"

#include <OpenSim/Simulation/Model/Analysis.h>
#include <OpenSim/Common/osimCommon.h>
#include <OpenSim/Simulation/osimSimulation.h>
#include <OpenSim/Actuators/osimActuators.h>

namespace OpenSim
{
  class ContactForceAnalysisTarget;

  class OSIMPLUGIN_API ContactForceAnalysis : public Analysis
  {
    OpenSim_DECLARE_CONCRETE_OBJECT(ContactForceAnalysis, Analysis);

    friend class ContactForceAnalysisTarget;

  public:
    ContactForceAnalysis(Model* aModel = nullptr);
    virtual ~ContactForceAnalysis();

    void setModel(Model& aModel) override;
    int begin(const SimTK::State& s) override;
    int step(const SimTK::State& s, int stepNumber) override;
    int end(const SimTK::State& s) override;
    int printResults(const std::string &aBaseName, const std::string &aDir, double aDT, const std::string &aExtension) override;

  protected:
    int record(const SimTK::State& s);

  private:
    void setNull();
    void constructProperties();
    void initialize(const SimTK::State& s);

  public:
    OpenSim_DECLARE_PROPERTY(consider_muscle_physiology, bool, "consider the effect of fiber length and shortening velocity on maximum muscle forces");
    OpenSim_DECLARE_PROPERTY(report_external_forces, bool, "If <true> all contacts in the model will be reported in OpenSim's file format for external forces.");
    OpenSim_DECLARE_PROPERTY(optimizer_convergence_tolerance, double, "convergence criterion for the optimizer");
    OpenSim_DECLARE_PROPERTY(optimizer_constraint_tolerance, double, "constraint tolerance for the optimizer");

  private:
    bool _initParameters;
    int _numActuators;
    int _numContactMultipliers;
    int _numContactConstraints;
    int _numQs;
    SimTK::Vector _parameters;
    SimTK::Vector _accelerations;
    std::vector<int> _contactPointBodies;
    std::vector<std::vector<int>> _contactPointBodiesToContactForces;
    Set<ContactPointOnPlane>* _contacts;
    Set<ScalarActuator>* _actuators;
    GCVSplineSet* _splines;
    Storage* _controlStorage;
    Storage* _forceStorage;
    Storage* _aggregatedExternalForcesStorage;
    Storage* _externalForcesStorage;
  };
}

#endif