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

#include "ContactPointOnPlane.h"

#include <OpenSim/OpenSim.h>

using namespace OpenSim;

ContactPointOnPlane::ContactPointOnPlane()
{
  this->setAuthors("Daniel Krueger, Felix Laufer");
  this->setReferences("D. Krüger and S. Wartzack, \"A contact model to simulate human-artifact interaction based on force optimization: "
                      "implementation and application to the analysis of a training machine\", "
                      "Computer Methods in Biomechanics and Biomedical Engineering, pp. 1-10, Oct. 2017, "
                      "DOI:10.1080/10255842.2017.1393804");

  constructProperty_optimal_force(100.0);
  constructProperty_static_friction(0.8);
  constructProperty_kinetic_friction(0.5);
  constructProperty_contact_tolerance(-1.0);
  constructProperty_contact_transition_zone(0.01);
  constructProperty_v_trans(0.1);
  constructProperty_plane_body("ground");
  constructProperty_plane_origin(SimTK::Vec3(0, 0, 0));
  constructProperty_plane_normal(SimTK::Vec3(0, 1, 0));
  constructProperty_point_body("ground");
  constructProperty_point_location(SimTK::Vec3(0, 0, 0));

  _minMultipliers.resize(getNumMultipliers());
  _minMultipliers[0] = 0; // normal component
  _minMultipliers[1] = -SimTK::Infinity; // shear 1
  _minMultipliers[2] = -SimTK::Infinity; // shear 2
  _maxMultipliers.resize(getNumMultipliers());
  _maxMultipliers = SimTK::Infinity; // all componentes
  _planeBody = nullptr;
  _pointBody = nullptr;
  _distance = 0;
  _velocity = 0;
}

double ContactPointOnPlane::computePotentialEnergy(const SimTK::State& s) const
{
  return 0;
}

void ContactPointOnPlane::generateDecorations(bool fixed, const ModelDisplayHints& hints, const SimTK::State& s, SimTK::Array_<SimTK::DecorativeGeometry>& appendToThis) const
{
  Super::generateDecorations(fixed, hints, s, appendToThis);

  // return if model not initialized
  if (!_model)
    return;

  if (_model->getMatterSubsystem().getStage(s) < SimTK::Stage::Position)
    return;

  // compute forces and points of application
  SimTK::Vec3 f_plane, f_point, p_plane, p_point;
  this->computeForceVectors(s, f_plane, f_point, p_plane, p_point);

  // scale forces so that optimal_force ~ 0.1m
  f_point = f_point / (10 * get_optimal_force());
  f_plane = f_plane / (10 * get_optimal_force());

  // draw force on point body
  SimTK::DecorativeLine line_point;
  line_point.setEndpoints(p_point, p_point - f_point);
  line_point.setColor(SimTK::Red);
  line_point.setLineThickness(3.0);
  appendToThis.push_back(line_point);

  // draw force on plane body
  SimTK::DecorativeLine line_plane;
  line_plane.setEndpoints(p_plane, p_plane - f_plane);
  line_plane.setColor(SimTK::Green);
  line_plane.setLineThickness(3.0);
  appendToThis.push_back(line_plane);
}

void ContactPointOnPlane::extendConnectToModel(Model& model)
{
  Super::extendConnectToModel(model);

  // check if body references exist in model
  std::string planeBody = get_plane_body();
  std::string pointBody = get_point_body();
  if (planeBody.find_last_of('/') != std::string::npos)
    planeBody = planeBody.substr(planeBody.find_last_of('/') + 1);
  if (pointBody.find_last_of('/') != std::string::npos)
    pointBody = pointBody.substr(pointBody.find_last_of('/') + 1);

  if (_model->updBodySet().contains(get_plane_body()))
    _planeBody = &_model->updBodySet().get(get_plane_body());
  else
  {
    if (_model->updGround().getName() == get_plane_body())
      _planeBody = &_model->updGround();
    else
    {
      _planeBody = nullptr;
      std::cerr << getName() + ": invalid plane_body (" + planeBody + ")" << std::endl;
    }
  }

  if (_model->updBodySet().contains(get_point_body()))
    _pointBody = &_model->updBodySet().get(get_point_body());
  else
  {
    _pointBody = nullptr;
    std::cerr << getName() + ": invalid point_body (" + pointBody + ")" << std::endl;
  }
}

void ContactPointOnPlane::extendAddToSystem(SimTK::MultibodySystem& system) const
{
  Super::extendAddToSystem(system);
  addDiscreteVariable("mult_normal", SimTK::Stage::Dynamics);
  addDiscreteVariable("mult_shear1", SimTK::Stage::Dynamics);
  addDiscreteVariable("mult_shear2", SimTK::Stage::Dynamics);
}

void ContactPointOnPlane::computeForce(const SimTK::State& s, SimTK::Vector_<SimTK::SpatialVec>& bodyForces, SimTK::Vector& generalizedForces) const
{
  // compute forces and points of application
  SimTK::Vec3 f_plane, f_point, p_plane, p_point;
  this->computeForceVectors(s, f_plane, f_point, p_plane, p_point);

  // apply forces to the bodies, applyForceToPoint requires the force application point to be in the body frame and the force vector itself to be in the ground frame
  const SimTK::Vec3 p_point_local = _model->getGround().findStationLocationInAnotherFrame(s, p_point, *_pointBody);
  const SimTK::Vec3 p_plane_local = _model->getGround().findStationLocationInAnotherFrame(s, p_plane, *_planeBody);
  applyForceToPoint(s, *_pointBody, p_point_local, f_point, bodyForces);
  applyForceToPoint(s, *_planeBody, p_plane_local, f_plane, bodyForces);
}

Array<std::string> ContactPointOnPlane::getRecordLabels() const
{
  Array<std::string> labels;
  labels.append(getName() + "_on_" + _pointBody->getName() + "_vx");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_vy");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_vz");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_px");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_py");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_pz");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_tx");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_ty");
  labels.append(getName() + "_on_" + _pointBody->getName() + "_tz");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_vx");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_vy");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_vz");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_px");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_py");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_pz");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_tx");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_ty");
  labels.append(getName() + "_on_" + _planeBody->getName() + "_tz");
  labels.append(getName() + "_dist");
  labels.append(getName() + "_vel");

  return labels;
}

Array<double> ContactPointOnPlane::getRecordValues(const SimTK::State& s) const
{
  // compute forces and points of application
  SimTK::Vec3 f_plane, f_point, p_plane, p_point;
  this->computeForceVectors(s, f_plane, f_point, p_plane, p_point);

  Array<double> values(0.0);
  values.append(f_point[0]);
  values.append(f_point[1]);
  values.append(f_point[2]);
  values.append(p_point[0]);
  values.append(p_point[1]);
  values.append(p_point[2]);
  values.append(0.0);
  values.append(0.0);
  values.append(0.0);
  values.append(f_plane[0]);
  values.append(f_plane[1]);
  values.append(f_plane[2]);
  values.append(p_plane[0]);
  values.append(p_plane[1]);
  values.append(p_plane[2]);
  values.append(0.0);
  values.append(0.0);
  values.append(0.0);
  values.append(_distance);
  values.append(_velocity);

  return values;
}

SimTK::Vector ContactPointOnPlane::getMaxMultipliers() const
{
  return _maxMultipliers;
}

SimTK::Vector ContactPointOnPlane::getMinMultipliers() const
{
  return _minMultipliers;
}

int ContactPointOnPlane::getNumMultipliers()
{
  return 3;
}

int ContactPointOnPlane::getNumMultiplierConstraints()
{
  return 4;
}

void ContactPointOnPlane::computeMultiplierConstraints(const SimTK::State& s, SimTK::Vector& constraints) const
{
  // compute constraints: limitation of shear components: shear<=friction*normal, constraints are independent from optimal_force (equal for all force components)
  // ToDo: maybe detect sliding regime and satisfy constraints
  SimTK::Vector multipliers = getMultipliers(s);
  const double mu = get_static_friction();
  // linearized friction cone (approximated by 4 planes)
  constraints[0] = multipliers[1] + mu * multipliers[0];
  constraints[1] = mu * multipliers[0] - multipliers[1];
  constraints[2] = multipliers[2] + mu * multipliers[0];
  constraints[3] = mu * multipliers[0] - multipliers[2];
}

void ContactPointOnPlane::computeMultiplierJacobian(const SimTK::State& s, SimTK::Matrix& jacobian) const
{
  // ToDo: maybe detect sliding regime and satisfy constraints
  const double mu = get_static_friction();
  // linearized friction cone
  jacobian = 0;
  jacobian[0][0] = mu;
  jacobian[0][1] = 1;
  jacobian[1][0] = mu;
  jacobian[1][1] = -1;
  jacobian[2][0] = mu;
  jacobian[2][2] = 1;
  jacobian[3][0] = mu;
  jacobian[3][2] = -1;
}

void ContactPointOnPlane::setOptimalForce(double force)
{
  set_optimal_force(force);
}

double ContactPointOnPlane::getOptimalForce() const
{
  return get_optimal_force();
}

void ContactPointOnPlane::setMultipliers(SimTK::State& s, const SimTK::Vector& multipliers) const
{
  setDiscreteVariableValue(s, "mult_normal", multipliers[0]);
  setDiscreteVariableValue(s, "mult_shear1", multipliers[1]);
  setDiscreteVariableValue(s, "mult_shear2", multipliers[2]);
}

SimTK::Vector ContactPointOnPlane::getMultipliers(const SimTK::State& s) const
{
  SimTK::Vector multipliers(getNumMultipliers());
  multipliers[0] = getDiscreteVariableValue(s, "mult_normal");
  multipliers[1] = getDiscreteVariableValue(s, "mult_shear1");
  multipliers[2] = getDiscreteVariableValue(s, "mult_shear2");  
  return multipliers;
}

void ContactPointOnPlane::computeForceVectors(const SimTK::State& s, SimTK::Vec3& f_on_plane_in_G, SimTK::Vec3& f_on_point_in_G, SimTK::Vec3& station_on_planeBody_in_G, SimTK::Vec3& station_on_pointBody_in_G) const
{
  if (!_model)
    return;

  // get contact multipliers stored in the state
  const SimTK::Vector multipliers = getMultipliers(s);

  // calculate force directions with respect to plane in ground frame
  const SimTK::UnitVec3 normal(_planeBody->expressVectorInGround(s, get_plane_normal()));

  // calculate arbitrary vector that is perpendicular to plane normal
  const SimTK::UnitVec3 shear1 = normal.perp();
  const SimTK::UnitVec3 shear2 = SimTK::UnitVec3(normal % shear1);

  // transform contact point into ground frame (=force station for pointBody)
  station_on_pointBody_in_G = _pointBody->findStationLocationInGround(s, get_point_location());

  // transform plane origin into ground frame
  const SimTK::Vec3 p_origin = _planeBody->findStationLocationInGround(s, get_plane_origin());

  // compute contact point to contact plane distance
  _distance = std::abs(~(station_on_pointBody_in_G - p_origin) * normal);
  
  //calculate orthogonal projection of the contact point onto the contact plane (=force station for planeBody)
  station_on_planeBody_in_G = station_on_pointBody_in_G - _distance * normal;

  // compute relative in-plane velocity of contact point
  const SimTK::Vec3 point_vel_in_G = _pointBody->getVelocityInGround(s)[1];
  const SimTK::Vec3 plane_vel_in_G = _planeBody->getVelocityInGround(s)[1];
  const SimTK::Vec3 v_rel_in_G = point_vel_in_G - plane_vel_in_G;

  // project v_rel into plane
  const SimTK::Vec3 v_rel_in_plane = v_rel_in_G - normal * (~v_rel_in_G * normal);
  _velocity = v_rel_in_plane.norm();

  // compute reference contact strength (depending on point to plane distance)
  double force_ref = 0;
  if (get_contact_tolerance() < 0) // negative values ==> distance has no effect
    force_ref = get_optimal_force();
  else if (_distance <= get_contact_tolerance() + get_contact_transition_zone()) // calculate force transition function
  {
    const double arg = (-2.0 * (_distance - get_contact_tolerance()) + get_contact_transition_zone()) / get_contact_transition_zone();
    force_ref = 0.5 * get_optimal_force() * (1 + std::tanh(arg * SimTK::Pi));
  }

  // detect if we are in the static or in the kinetic regime, if so: caculate force on plane body (normal component in opposite direction of plane normal!)
  if (_velocity <= get_v_trans())
    f_on_plane_in_G = (-multipliers[0] * normal + multipliers[1] * shear1 + multipliers[2] * shear2) * force_ref;
  else // otherwise: kinetic regime (shear multipliers have no effect)
  {
    // shear force in opposite direction of in-plane velocity
    const SimTK::Vec3 shear = multipliers[0] * get_kinetic_friction() * v_rel_in_plane.normalize();
    // total force
    f_on_plane_in_G = (-multipliers[0] * normal + shear) * force_ref;
  }

  // force on point body acts along the opposite direction
  f_on_point_in_G = -f_on_plane_in_G;
}