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

#ifndef _OSIM_CONTACT_POINT_ON_PLANE_H_
#define _OSIM_CONTACT_POINT_ON_PLANE_H_

#include "osimPluginDLL.h"

#include <OpenSim/Common/PropertyStr.h>
#include <OpenSim/Common/PropertyDbl.h>
#include <OpenSim/Simulation/Model/Force.h>
#include <Simulation/SimbodyEngine/Body.h>

namespace OpenSim
{
  class OSIMPLUGIN_API ContactPointOnPlane : public Force
  {
    OpenSim_DECLARE_CONCRETE_OBJECT(ContactPointOnPlane, Force);

  public:
    ContactPointOnPlane();

    void computeForce(const SimTK::State& s, SimTK::Vector_<SimTK::SpatialVec>& bodyForces, SimTK::Vector& generalizedForces) const override;
    double computePotentialEnergy(const SimTK::State& s) const override;
    void generateDecorations(bool fixed, const ModelDisplayHints& hints, const SimTK::State& s, SimTK::Array_<SimTK::DecorativeGeometry>& appendToThis) const override;

    Array<std::string> getRecordLabels() const override;
    Array<double> getRecordValues(const SimTK::State& s) const override;

    SimTK::Vector getMultipliers(const SimTK::State& s) const;
    void setMultipliers(SimTK::State& s, const SimTK::Vector& multipliers) const;

    SimTK::Vector getMaxMultipliers() const;
    SimTK::Vector getMinMultipliers() const;

    void setOptimalForce(double force);
    double getOptimalForce() const;

    static int getNumMultipliers();
    static int getNumMultiplierConstraints();

    void computeMultiplierConstraints(const SimTK::State& s, SimTK::Vector& constraints) const;
    void computeMultiplierJacobian(const SimTK::State& s, SimTK::Matrix& jacobian) const;

  protected:
    void extendConnectToModel(Model& model) override;
    void extendAddToSystem(SimTK::MultibodySystem& system) const override;

  private:
    void computeForceVectors(const SimTK::State& s, SimTK::Vec3& f_on_plane_in_G, SimTK::Vec3& f_on_point_in_G, SimTK::Vec3& station_on_planeBody_in_G, SimTK::Vec3& station_on_pointBody_in_G) const;

  public:
    OpenSim_DECLARE_PROPERTY(optimal_force, double, "optimal force of the contact (reference)");
    OpenSim_DECLARE_PROPERTY(static_friction, double, "static friction coefficient, default: 0.8");
    OpenSim_DECLARE_PROPERTY(kinetic_friction, double, "kinetic friction coefficient, default: 0.5");
    OpenSim_DECLARE_PROPERTY(contact_tolerance, double, "tolerance on point to plane distance, default: -1 (no effect)");
    OpenSim_DECLARE_PROPERTY(contact_transition_zone, double, "size of the contact transition zone, default: 0.01");
    OpenSim_DECLARE_PROPERTY(v_trans, double, "transition velocity (static->kinetic regime), default: 0.1");
    OpenSim_DECLARE_PROPERTY(plane_body, std::string, "name of the plane body");
    OpenSim_DECLARE_PROPERTY(plane_origin, SimTK::Vec3, "origin of the plane in plane body");
    OpenSim_DECLARE_PROPERTY(plane_normal, SimTK::Vec3, "normal of the plane in plane body");
    OpenSim_DECLARE_PROPERTY(point_body, std::string, "name of the point body");
    OpenSim_DECLARE_PROPERTY(point_location, SimTK::Vec3, "location of the point in point body");

  protected:
    SimTK::Vector _maxMultipliers;
    SimTK::Vector _minMultipliers;
    PhysicalFrame* _planeBody;
    PhysicalFrame* _pointBody;
    mutable double _distance, _velocity;
  };
}

#endif