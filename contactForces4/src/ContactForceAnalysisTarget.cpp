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

#include "ContactForceAnalysisTarget.h"

using namespace OpenSim;

ContactForceAnalysisTarget::ContactForceAnalysisTarget(ContactForceAnalysis* analysis)
{
  _analysis = analysis;
  _numResiduals = 0;
  _systemMass = 0;
}

ContactForceAnalysisTarget::~ContactForceAnalysisTarget()
{}

void ContactForceAnalysisTarget::initialize(const SimTK::State& s)
{
  // set the internal state
  _si = s;

  // determine free/unconstrained coordinates
  for (int q = 0; q < _analysis->_model->getCoordinateSet().getSize(); q++)
  {
    const Coordinate& coord = _analysis->_model->getCoordinateSet().get(q);
    // check if coordinate is locked or prescribed => exclude from acceleration constraints
    if (!coord.isConstrained(_si))
      _qs2FreeQs.push_back(q);
  }
  _numResiduals = _qs2FreeQs.size();

  // disable all controllers in the model 
  _analysis->_model->setAllControllersEnabled(false);

  // compute optimal forces for actuators
  _analysis->_model->getMultibodySystem().realize(_si, SimTK::Stage::Velocity);
  _optimalForces.resize(_analysis->_numActuators);
  for (int m = 0; m < _analysis->_numActuators; ++m)
  {
    double force;
    // try to cast into muscle
    Muscle* muscle = dynamic_cast<Muscle*>(&_analysis->_actuators->get(m));
    if (muscle)
    {
      if (_analysis->get_consider_muscle_physiology()) // get max force from hill-model
        force = muscle->calcInextensibleTendonActiveFiberForce(_si, 1.0);
      else // use max isometric peak force
        force = muscle->getMaxIsometricForce();
    }
    else // simple actuator 
      force = _analysis->_actuators->get(m).getOptimalForce();

    _optimalForces[m] = force;
  }

  // set the structure of the problem    
  this->setNumParameters(_analysis->_numActuators + _analysis->_numContactMultipliers);

  // vectors for lower and upper parameter bounds
  SimTK::Vector lb(getNumParameters()), ub(getNumParameters());

  // set actuator control limits
  for (int c = 0; c < _analysis->_numActuators; ++c)
  {
    lb[c] = _analysis->_actuators->get(c).getMinControl();
    ub[c] = _analysis->_actuators->get(c).getMaxControl();
  }

  // set contact multipliers limits
  const int numMult = ContactPointOnPlane::getNumMultipliers();
  int row_offset = _analysis->_numActuators;
  for (int c = 0; c < _analysis->_contacts->getSize(); ++c)
  {
    lb(row_offset, numMult) = _analysis->_contacts->get(c).getMinMultipliers();
    ub(row_offset, numMult) = _analysis->_contacts->get(c).getMaxMultipliers();
    row_offset += numMult;
  }
  this->setParameterLimits(lb, ub);

  // set number of equality constraints= residual conditions
  this->setNumEqualityConstraints(_numResiduals);
  this->setNumLinearEqualityConstraints(_numResiduals);
  // set number of inequality constraints = linear inequalities on contact multipliers
  this->setNumInequalityConstraints(_analysis->_numContactConstraints);
  this->setNumLinearInequalityConstraints(_analysis->_numContactConstraints);

  // calculate system mass force normalization factor
  _systemMass = 0;
  for (int b = 0; b < _analysis->_model->getBodySet().getSize(); ++b)
    _systemMass += _analysis->_model->getBodySet().get(b).getMass();

  // multiply by gravity
  _systemMass = _analysis->_model->getGravity().norm();

  // compute constant constraint vector
  SimTK::Vector params(getNumParameters(), 0.0);
  _constraintVector.resize(getNumConstraints());
  this->computeConstraints(params, _constraintVector);

  // compute linear constraint matrix
  _constraintMatrix.resize(getNumConstraints(), getNumParameters());
  const SimTK::Vector constraints(getNumConstraints());
  // loop through controls
  for (int p = 0; p < this->_analysis->_numActuators + _analysis->_numContactMultipliers; ++p)
  {
    params[p] = 1;
    this->computeConstraints(params, constraints);
    //compute p-th column of constraint matrix
    _constraintMatrix.updCol(p) = constraints - _constraintVector;
    params[p] = 0;
  }
}

SimTK::Vector ContactForceAnalysisTarget::initParameters() const
{
  SimTK::Vector parameters(getNumParameters());
  parameters = 0;
  // ToDo: initialize to meaningful values e.g. between lb and ub.
  return parameters;
}

int ContactForceAnalysisTarget::constraintFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Vector& errors) const
{
  errors = _constraintMatrix * parameters + _constraintVector;
  return 0;
}

int ContactForceAnalysisTarget::constraintJacobian(const SimTK::Vector& parameters, bool new_parameters, SimTK::Matrix& jacobian) const
{
  jacobian = _constraintMatrix;
  return 0;
}

int ContactForceAnalysisTarget::objectiveFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Real& objective) const
{
  objective = ~parameters * parameters;
  return 0;
}

int ContactForceAnalysisTarget::gradientFunc(const SimTK::Vector& parameters, bool new_parameters, SimTK::Vector& gradient) const
{
  gradient = 2 * parameters;
  return 0;
}

void ContactForceAnalysisTarget::setForces(const SimTK::Vector& parameters, SimTK::State& s)
{
  // set forces for Actuators
  for (int m = 0; m < _analysis->_numActuators; ++m)
  {
    // override actuator force
    _analysis->_actuators->get(m).setOverrideActuation(s, _optimalForces[m] * parameters[m]);
    _analysis->_actuators->get(m).overrideActuation(s, true);
  }

  // set contact multipliers
  int mult_idx = _analysis->_numActuators;
  for (int c = 0; c < _analysis->_contacts->getSize(); ++c)
  {
    const int numMult = _analysis->_contacts->get(c).getNumMultipliers();
    _analysis->_contacts->get(c).setMultipliers(s, parameters(mult_idx, numMult));
    mult_idx += numMult;
  }
}

SimTK::String ContactForceAnalysisTarget::printPerformance(const SimTK::Vector& parameters) const
{
  double objective;
  SimTK::Vector errors(getNumConstraints());
  objectiveFunc(parameters, true, objective);
  constraintFunc(parameters, true, errors);
  std::stringstream ss;
  ss << "performance=" << objective << " constraint violation=" << sqrt(~errors(0, _numResiduals) * errors(0, _numResiduals));
  return ss.str();
}

void ContactForceAnalysisTarget::computeConstraints(const SimTK::Vector& parameters, const SimTK::Vector& constraints)
{
  const SimTK::MultibodySystem& system = _analysis->_model->getMultibodySystem();

  // set forces for actuators and contacts:
  setForces(parameters, _si);

  // compute generalized accelerations
  system.realize(_si, SimTK::Stage::Acceleration);
  SimTK::Vector r = _si.getQDotDot();

  // return only the residuals corresponding to the free Qs
  for (int a = 0; a < _numResiduals; ++a)
    constraints(0, _numResiduals)[a] = (r[_qs2FreeQs[a]] - _analysis->_accelerations[_qs2FreeQs[a]]) / _systemMass;

  // evaluate contact constraints
  int idx = _numResiduals;
  for (int c = 0; c < _analysis->_contacts->getSize(); ++c)
  {
    const int numCons = ContactPointOnPlane::getNumMultiplierConstraints();
    _analysis->_contacts->get(c).computeMultiplierConstraints(_si, constraints(idx, numCons));
    idx += numCons;
  }
}