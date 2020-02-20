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

#include "ContactForceAnalysis.h"
#include "ContactForceAnalysisTarget.h"

using namespace OpenSim;

ContactForceAnalysis::ContactForceAnalysis(Model* aModel) : Analysis(aModel)
{
  this->setAuthors("Daniel Krueger, Felix Laufer");
  this->setReferences("D. Krüger and S. Wartzack, \"A contact model to simulate human-artifact interaction based on force optimization: "
    "implementation and application to the analysis of a training machine\", "
    "Computer Methods in Biomechanics and Biomedical Engineering, pp. 1-10, Oct. 2017, "
    "DOI:10.1080/10255842.2017.1393804");

  setNull();
  constructProperties();

  if (aModel)
    ContactForceAnalysis::setModel(*aModel);
}

ContactForceAnalysis::~ContactForceAnalysis()
{
  if (_contacts != nullptr)
  {
    delete _contacts;
    _contacts = nullptr;
  }

  if (_actuators != nullptr)
  {
    delete _actuators;
    _actuators = nullptr;
  }

  if (_splines != nullptr)
  {
    delete _splines;
    _splines = nullptr;
  }

  if (_controlStorage != nullptr)
  {
    delete _controlStorage;
    _controlStorage = nullptr;
  }

  if (_forceStorage != nullptr)
  {
    delete _forceStorage;
    _forceStorage = nullptr;
  }

  if (_aggregatedExternalForcesStorage != nullptr)
  {
    delete _aggregatedExternalForcesStorage;
    _aggregatedExternalForcesStorage = nullptr;
  }

  if (_externalForcesStorage != nullptr)
  {
    delete _externalForcesStorage;
    _externalForcesStorage = nullptr;
  }
}

void ContactForceAnalysis::setNull()
{
  _contacts = nullptr;
  _actuators = nullptr;
  _contactPointBodies.clear();
  _contactPointBodiesToContactForces.clear();
  _splines = nullptr;
  _parameters.resize(0);
  _accelerations.resize(0);
  _controlStorage = nullptr;
  _forceStorage = nullptr;
  _externalForcesStorage = nullptr;
  _aggregatedExternalForcesStorage = nullptr;
  _initParameters = true;
  _numActuators = 0;
  _numContactMultipliers = 0;
  _numContactConstraints = 0;
  _numQs = 0;

  setName("ContactForceAnalysis");
}

void ContactForceAnalysis::constructProperties()
{
  constructProperty_consider_muscle_physiology(false);
  constructProperty_report_external_forces(true);
  constructProperty_optimizer_convergence_tolerance(1e-4);
  constructProperty_optimizer_constraint_tolerance(1e-6);
}

void ContactForceAnalysis::setModel(Model& aModel)
{
  Super::setModel(aModel);
}

void ContactForceAnalysis::initialize(const SimTK::State& s)
{
  this->setNull();
  _numQs = _model->getCoordinateSet().getSize();
  _accelerations.resize(_numQs);
  _accelerations = 0;

  // analyze all forces in the model, extract actuators and contacts
  ForceSet& forces = _model->updForceSet();

  _contacts = new Set<ContactPointOnPlane>();
  _actuators = new Set<ScalarActuator>();
  _contacts->setMemoryOwner(false);
  _actuators->setMemoryOwner(false);

  for (int f = 0; f < forces.getSize(); ++f)
  {
    Force& force = forces.get(f);

    // skip disabled forces
    if (!force.appliesForce(s))
      continue;

    // try to cast into Actuator
    ScalarActuator* act = dynamic_cast<ScalarActuator*>(&force);
    if (act)
      _actuators->adoptAndAppend(act);

    // try to cast into contact
    ContactPointOnPlane* cont = dynamic_cast<ContactPointOnPlane*>(&force);
    if (cont)
      _contacts->adoptAndAppend(cont);
  }

  // save number of actuators
  _numActuators = _actuators->getSize();

  // ask all contact forces for their number of multipliers and constraints
  _numContactMultipliers = 0;
  _numContactConstraints = 0;

  for (int cf = 0; cf < _contacts->getSize(); ++cf)
  {
    _numContactMultipliers += ContactPointOnPlane::getNumMultipliers();
    _numContactConstraints += ContactPointOnPlane::getNumMultiplierConstraints();
  }

  // get the column labels for all controllable forces (actuators and contact multipliers)
  Array<std::string> labels;
  labels.append("time");

  for (int a = 0; a < _numActuators; ++a)
    labels.append(_actuators->get(a).getName());

  for (int c = 0; c < _contacts->getSize(); ++c)
  {
    const ContactPointOnPlane& contact = _contacts->get(c);
    const Array<std::string> records = contact.getRecordLabels();
    // caution: we assume that contacts report the force vector in the first 3 records
    for (int m = 0; m < 3; ++m)
      labels.append(records[m]);
  }

  // add kinematics
  for (int q = 0; q < _model->getCoordinateSet().getSize(); ++q)
    labels.append(_model->getCoordinateSet().get(q).getName());

  // initialize control storage
  _controlStorage = new Storage(1000, _model->getName() + "_controls");
  _controlStorage->setDescription("control values for actuators and contact elements");
  _controlStorage->setColumnLabels(labels);

  // initialize force storage
  _forceStorage = new Storage(1000, _model->getName() + "_forces");
  _forceStorage->setDescription("force values for actuators and contact elements");
  _forceStorage->setColumnLabels(labels);

  // build splines for kinematics
  if (_splines != nullptr)
  {
    delete _splines;
    _splines = nullptr;
  }
  _splines = new GCVSplineSet(5, this->_statesStore);

  // functions must correspond to model coordinates in the correct order
  for (int q = 0; q < _numQs; ++q)
  {
    const Coordinate& coord = _model->getCoordinateSet().get(q);
    const std::string coordAbsPathStringValue = coord.getAbsolutePathString() + "/value";
    if (_splines->contains(coordAbsPathStringValue))
      _splines->insert(q, _splines->get(coordAbsPathStringValue));
    else
    {
      _splines->insert(q, new Constant(coord.getDefaultValue()));
      std::cout << "ContactForceAnalysis: state storage does not contain coordinate " << coord.getName() << ", assuming default value" << std::endl;
    }
  }

  // cut off, remaining columns e.g. speed values or additional states
  if (_splines->getSize() > _numQs)
    _splines->setSize(_numQs);

  // map contact forces to bodies
  if (get_report_external_forces())
  {
    // loop through contact forces
    for (int c = 0; c < _contacts->getSize(); ++c)
    {
      // try to cast into Contact_PointOnPlane
      ContactPointOnPlane* pc = dynamic_cast<ContactPointOnPlane*>(&_contacts->get(c));
      if (pc)
      {
        // get point BodyName
        const std::string pointBodyName = pc->getProperty_point_body().getValue();
        // retrieve index of OpenSim Body
        int bodyIndex = _model->getBodySet().getIndex(pointBodyName);
        if (bodyIndex != -1)
        {
          // look if the body is not already in the vector
          std::vector<int>::iterator it = std::find(_contactPointBodies.begin(), _contactPointBodies.end(), bodyIndex);
          if (it == _contactPointBodies.end())
          {
            // append the index
            _contactPointBodies.push_back(bodyIndex);
            // allocate new slot in mapping vector and store force index
            std::vector<int> slot;
            slot.push_back(c);
            _contactPointBodiesToContactForces.push_back(slot);
          }
          else
          {
            // body and mapping slot already existent ==> just add force index to the corresponding slot
            const int idx = it - _contactPointBodies.begin();
            _contactPointBodiesToContactForces[idx].push_back(c);
          }
        }
      }
    }

    //get the column labels for all contactPoint bodies and individual contact forces
    Array<std::string> bodylabels, contactlabels;
    bodylabels.append("time");
    contactlabels.append("time");

    for (int b = 0; b < _contactPointBodies.size(); ++b)
    {
      const std::string bodyName = _model->getBodySet().get(_contactPointBodies[b]).getName();

      // construct labels for force vector, torque and application point in ground
      bodylabels.append(bodyName + "_force_vx");
      bodylabels.append(bodyName + "_force_vy");
      bodylabels.append(bodyName + "_force_vz");
      bodylabels.append(bodyName + "_force_px");
      bodylabels.append(bodyName + "_force_py");
      bodylabels.append(bodyName + "_force_pz");
      bodylabels.append(bodyName + "_torque_x");
      bodylabels.append(bodyName + "_torque_y");
      bodylabels.append(bodyName + "_torque_z");
    }

    // construct labels for individual forces
    for (int c = 0; c < _contacts->getSize(); ++c)
    {
      Array<std::string> labels = _contacts->get(c).getRecordLabels();
      // strip array to 18 mandatory fields
      for (int l = 0; l < 18; ++l)
        contactlabels.append(labels[l]);
    }

    // initialize storages
    _aggregatedExternalForcesStorage = new Storage(1000, _model->getName() + "_externalForces_aggregated");
    _aggregatedExternalForcesStorage->setDescription("aggregated external forces from PointOnPlaneContacts");
    _aggregatedExternalForcesStorage->setColumnLabels(bodylabels);

    _externalForcesStorage = new Storage(1000, _model->getName() + "_externalForces");
    _externalForcesStorage->setDescription("external forces from conditional contacts");
    _externalForcesStorage->setColumnLabels(contactlabels);
  }

  std::cout << "init successful" << std::endl;
  std::cout << "number of bodies with contacts " << _contactPointBodies.size() << std::endl;
}

int ContactForceAnalysis::record(const SimTK::State& s)
{
  // temporary storage for all values to be reported to the storage
  Array<double> controls, forces;
  controls.setSize(0);
  forces.setSize(0);
  // copy the state since it may be necessary to modify it
  SimTK::State s_temp = s;
  const double time = s_temp.getTime();

  // get kinematics from splines
  for (int q = 0; q < _numQs; ++q)
  {
    s_temp.updQ()[q] = _splines->evaluate(q, 0, time);
    s_temp.updQDot()[q] = _splines->evaluate(q, 1, time);
    _accelerations[q] = _splines->evaluate(q, 2, time);
  }

  _model->getMultibodySystem().realize(s_temp, SimTK::Stage::Velocity);

  // create new static optimization problem
  ContactForceAnalysisTarget* problem = new ContactForceAnalysisTarget(this);
  problem->initialize(s_temp);

  // create optimizer
  const SimTK::OptimizerAlgorithm algorithm = SimTK::OptimizerAlgorithm::InteriorPoint;
  SimTK::Optimizer* optimizer = new SimTK::Optimizer(*problem, algorithm);
  optimizer->setConvergenceTolerance(get_optimizer_convergence_tolerance());
  optimizer->setConstraintTolerance(get_optimizer_constraint_tolerance());
  optimizer->setMaxIterations(1000);
  optimizer->useNumericalGradient(false);
  optimizer->useNumericalJacobian(false);
  optimizer->setDiagnosticsLevel(2);
  optimizer->setLimitedMemoryHistory(500);
  optimizer->setAdvancedBoolOption("warm_start", true);
  optimizer->setAdvancedRealOption("obj_scaling_factor", 1);
  optimizer->setAdvancedStrOption("nlp_scaling_method", "none");

  // try to solve
  try
  {
    SimTK::Vector temp_params;
    if (_initParameters)
      temp_params = problem->initParameters();
    else
      temp_params = _parameters;

    optimizer->optimize(temp_params);
    // success => save parameters and re-use them as a starting guess in the next iteration
    _parameters = temp_params;
    _initParameters = false;
    std::cout << "time=" << time << " " << problem->printPerformance(_parameters) << std::endl;
  }
  catch (SimTK::Exception::OptimizerFailed& e)
  {
    // something went wrong
    std::cout << "time=" << time << " optimization failed!" << std::endl;
    _initParameters = true;
    _parameters = problem->initParameters();
  }

  // store controls
  for (int p = 0; p < _parameters.size(); ++p)
    controls.append(_parameters[p]);

  // evaluate forces and store them
  problem->setForces(_parameters, s_temp);
  _model->getMultibodySystem().realize(s_temp, SimTK::Stage::Velocity);
  _model->getMultibodySystem().realize(s_temp, SimTK::Stage::Dynamics);

  for (int a = 0; a < _numActuators; ++a)
    forces.append(_actuators->get(a).getActuation(s_temp));

  for (int c = 0; c < _contacts->getSize(); ++c)
  {
    const Array<double> records = _contacts->get(c).getRecordValues(s_temp);
    // caution, we assume that contacts report their force vector within the first 3 record entries!
    for (int r = 0; r < 3; ++r)
      forces.append(records[r]);
  }

  // evaluate kinematics
  for (int q = 0; q < _model->getCoordinateSet().getSize(); ++q)
    forces.append(_model->getCoordinateSet().get(q).getValue(s_temp));

  // put values into storages
  _controlStorage->append(s_temp.getTime(), controls.getSize(), &controls[0]);
  _forceStorage->append(s_temp.getTime(), forces.getSize(), &forces[0]);

  // report external forces if required
  if (get_report_external_forces())
  {
    // 1. report record values of individual contacts
    Array<double> contactForces;
    contactForces.setSize(0);
    for (int c = 0; c < _contacts->getSize(); ++c)
    {
      // get the record values
      const Array<double> records = _contacts->get(c).getRecordValues(s_temp);
      // report 18 mandatory fields
      for (int l = 0; l < 18; ++l)
        contactForces.append(records[l]);
    }

    // 2. aggregate Point on Plane Contacts to Point Bodies
    Array<double> externalForces;
    externalForces.setSize(0);

    // loop through all bodies with contact points
    for (int b = 0; b < _contactPointBodies.size(); ++b)
    {
      // compute resulting force and torque (with respect to ground) of contact forces
      SimTK::Vec3 torque, force, cop;
      torque = 0;
      force = 0;
      cop = 0;
      double sum_norm = 0;
      for (int f = 0; f < _contactPointBodiesToContactForces[b].size(); ++f)
      {
        // get corresponding contact force
        const ContactPointOnPlane& contact = _contacts->get(_contactPointBodiesToContactForces[b][f]);
        // get record values
        const Array<double> records = contact.getRecordValues(s_temp);
        // force in ground
        const SimTK::Vec3 f_in_G = SimTK::Vec3(records[0], records[1], records[2]);
        // point of application in ground
        const SimTK::Vec3 p_in_G = SimTK::Vec3(records[3], records[4], records[5]);
        // calculate a kind of pseudo center of pressure: cop=sum(point_of_application*norm(force)/sum(norm(force)))!
        cop += p_in_G * f_in_G.norm();
        sum_norm += f_in_G.norm();
        // calculate torque from current force with respect to ground origin (cross product)
        torque += cross(f_in_G, p_in_G);
        // add force contribution to resulting force
        force += f_in_G;
      }

      //save resulting force vector
      externalForces.append(force[0]);
      externalForces.append(force[1]);
      externalForces.append(force[2]);

      // calculate point of application (a pseudo center of pressure)
      if (sum_norm > 1e-6)
        cop *= 1.0 / sum_norm;
      else
        cop *= 1.0 / _contactPointBodiesToContactForces[b].size(); // avoid division by zero if all contacts are switched off

      // save resulting point
      externalForces.append(cop[0]);
      externalForces.append(cop[1]);
      externalForces.append(cop[2]);

      // compute compensation torque effect on body
      torque = torque - cross(force, cop);
      externalForces.append(torque[0]);
      externalForces.append(torque[1]);
      externalForces.append(torque[2]);
    }

    // store results
    _aggregatedExternalForcesStorage->append(s_temp.getTime(), externalForces.getSize(), &externalForces[0]);
    _externalForcesStorage->append(s_temp.getTime(), contactForces.getSize(), &contactForces[0]);
  }

  delete optimizer;
  delete problem;
  return 0;
}

int ContactForceAnalysis::begin(const SimTK::State& s)
{
  if (!proceed())
    return 0;

  initialize(s);

  std::cout << "ContactForceAnalysis started" << std::endl;
  if (get_consider_muscle_physiology())
    std::cout << "-considering muscle physiology" << std::endl;

  _controlStorage->reset(s.getTime());
  _forceStorage->reset(s.getTime());

  if (get_report_external_forces())
  {
    _aggregatedExternalForcesStorage->reset(s.getTime());
    _externalForcesStorage->reset(s.getTime());
  }

  return 0;
}

int ContactForceAnalysis::step(const SimTK::State& s, int stepNumber)
{
  if (!proceed(stepNumber))
    return 0;

  record(s);

  return 0;
}

int ContactForceAnalysis::end(const SimTK::State& s)
{
  if (!proceed())
    return 0;

  record(s);

  return 0;
}

int ContactForceAnalysis::printResults(const std::string &aBaseName, const std::string &aDir, double aDT, const std::string &aExtension)
{
  const std::string controlfile = aBaseName + "_" + getName() + "_controls";
  const std::string forcefile = aBaseName + "_" + getName() + "_forces";
  const std::string aggregatedforcefile = aBaseName + "_" + getName() + "_externalForces_aggregated";
  const std::string externalforcefile = aBaseName + "_" + getName() + "_externalForces";

  // save force and control storages
  Storage::printResult(_controlStorage, controlfile, aDir, aDT, aExtension);
  Storage::printResult(_forceStorage, forcefile, aDir, aDT, aExtension);

  if (get_report_external_forces())
  {
    Storage::printResult(_aggregatedExternalForcesStorage, aggregatedforcefile, aDir, aDT, aExtension);
    Storage::printResult(_externalForcesStorage, externalforcefile, aDir, aDT, aExtension);

    // create External loads file for individual contact forces
    ExternalLoads extLoads;
    extLoads.connectToModel(*_model);
    extLoads.setDataFileName(externalforcefile + aExtension);
    // loop through contact forces
    for (int c = 0; c < _contacts->getSize(); c++)
    {
      // obtain record labels
      const Array<std::string> labels = _contacts->get(c).getRecordLabels();
      std::string body1_name, body2_name;

      // try to find the name of a valid body in the force identifier
      const std::string gName = _model->getGround().getName();
      if (labels[0].find(gName) != std::string::npos)
        body1_name = gName;
      if (labels[9].find(gName) != std::string::npos)
        body2_name = gName;

      for (int b = 0; b < _model->getBodySet().getSize(); ++b)
      {
        const std::string bName = _model->getBodySet().get(b).getName();
        if (labels[0].find(bName) != std::string::npos)
          body1_name = bName;
        if (labels[9].find(bName) != std::string::npos)
          body2_name = bName;
      }

      if (!body1_name.empty() && !body2_name.empty())
      {
        // we found the referenced bodies and create 2 external force objects
        ExternalForce* force1 = new ExternalForce();
        force1->setAppliedToBodyName(body1_name);
        force1->setForceIdentifier(labels[0].substr(0, labels[0].length() - 1));
        force1->setForceExpressedInBodyName("ground");
        force1->setPointIdentifier(labels[3].substr(0, labels[3].length() - 1));
        force1->setPointExpressedInBodyName("ground");
        force1->setName(_contacts->get(c).getName() + "_on_" + body1_name);
        force1->setTorqueIdentifier(labels[6].substr(0, labels[6].length() - 1));
        extLoads.adoptAndAppend(force1);

        ExternalForce* force2 = new ExternalForce();
        force2->setAppliedToBodyName(body2_name);
        force2->setForceIdentifier(labels[9].substr(0, labels[9].length() - 1));
        force2->setForceExpressedInBodyName("ground");
        force2->setPointIdentifier(labels[12].substr(0, labels[12].length() - 1));
        force2->setPointExpressedInBodyName("ground");
        force2->setName(_contacts->get(c).getName() + "_on_" + body2_name);
        force2->setTorqueIdentifier(labels[15].substr(0, labels[15].length() - 1));
        extLoads.adoptAndAppend(force2);
      }
    }

    extLoads.print(aDir + "\\" + aBaseName + "_" + getName() + "_externalForces.xml");
  }

  return 0;
}