/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

/* ----------------------------------------------------------------------
   Contributing author: Connor (May2023)

------------------------------------------------------------------------- */
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "core/Atoms.h"
#include "tools/Tools.h"
#include "tools/Angle.h"
#include "tools/IFile.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>
#include <string>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>      // std::iota; equivalent to range() in python
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

std::vector<float> tensor_to_vector(const torch::Tensor& x) {
    return std::vector<float>(x.data<float>(), x.data<float>() + x.numel());
}

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR GRAPH
/*

This file is used for post-processing. It outputs number of nodes being classified into each class. \
Several inputs need to be provided. ATOMS corresponds to the position of nodes which \
is needed for neighborhood definitions and corresponding kNN algorithm. KCUT is the number of nearest neighbor \
for kNN algorithm. MODEL takes a frozen pytorch model as input.

\plumedfile 
LOAD FILE=Graph.cpp

# Define groups for the CV
C: GROUP ATOMS=1-285

GRAPH ...
 LABEL=model
 ATOMS=C
 KCUT=50
 MODEL=model-local.pt
... GRAPH

PRINT STRIDE=1  ARG=* FILE=COLVAR
\endplumedfile

*/
//+ENDPLUMEDOC

class GraphGenerator: 
  public Colvar
{
  bool serial,pbc;
  vector<AtomNumber> atom_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  unsigned kcut, _n_out, nat, _n_feat;
  vector<size_t> sort_indexes(const vector<float> &v); // declare argsort function
  torch::jit::script::Module _model;
    
public:
  explicit GraphGenerator(const ActionOptions&);
  void calculate();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(GraphGenerator,"GRAPH")

void GraphGenerator::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial");
  keys.add("atoms","ATOMS","Reference particles for constructing graphs");
  keys.add("compulsory","KCUT","50","k-nearest neighbor for creating neighorlist"); // 
  keys.add("optional","MODEL","filename of the trained model"); 
  keys.addOutputComponent("node", "default", "NN outputs"); 

}

GraphGenerator::GraphGenerator(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false)
{  
  parseFlag("SERIAL",serial);
  parseAtomList("ATOMS", atom_lista); // list of node 
  
  kcut = 1; 
  parse("KCUT", kcut); // read knn parameter 
  if( kcut<2 ) error("Please set KCUT larger than 1\n");
  log.printf("  The neighbor is defined as %f nearest neighbors\n", kcut);
      
  //parse model name
  std::string fname="model.pt";
  parse("MODEL",fname); 
      
  //deserialize the model from file
  try {
    _model = torch::jit::load(fname);
  }
  catch (const c10::Error& e) {
    error("Cannot find Pytorch model.");    
  }
        
  checkRead();

  nat = atom_lista.size();

  _n_feat = nat * kcut; 
  log.printf("Number of neighbors: %d \n",kcut);
      
  //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<int> input_test1 ( _n_feat ); // edge index vector
  std::vector<float> input_test2 ( _n_feat ); // edge feature vector
  std::vector<float> input_test3 ( nat ); // node feature vector
      
  torch::Tensor single_input1 = torch::tensor(input_test1, torch::dtype(torch::kInt64)).view(-1);  
  torch::Tensor single_input2 = torch::tensor(input_test2).view(-1);
  torch::Tensor single_input3 = torch::tensor(input_test3).view({-1,1});  
  
  std::vector<torch::jit::IValue> inputs;
  
  inputs.push_back( single_input3 );
  inputs.push_back( single_input1 );
  inputs.push_back( single_input2 );
  torch::Tensor output = _model.forward( inputs ).toTensor(); 
  vector<float> cvs = tensor_to_vector (output);
  _n_out=cvs.size();

  //create components
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }
  
  //print log
  log.printf("Number of outputs: %d \n",_n_out);
      
  atomsToRequest.reserve ( atom_lista.size() ); // Initilization of atoms list 
  atomsToRequest.insert (atomsToRequest.end(), atom_lista.begin(), atom_lista.end() );
  requestAtoms(atomsToRequest);  
}

// generator
void GraphGenerator::calculate()
{
	
  if(pbc) makeWhole();
  // Setup parallelization
  unsigned stride = comm.Get_size();
  unsigned rank = comm.Get_rank();
  if(serial){
    stride = 1;
    rank = 0;
  } else {
    stride = comm.Get_size();
    rank = comm.Get_rank(); 
  }
  
  Matrix<double> dist_mat(nat, kcut);
  Matrix<double> idxn_mat(nat, kcut);
  Matrix<Vector> dist_deriv_mat(nat, kcut);
  
  for(unsigned i=rank;i<nat;i+=stride) {
     
    vector<float> dist_array(nat);
    vector<Vector> der_dist_array(nat);
      
    for(unsigned j=0;j<nat;j+=1) {
      Vector distance;
      distance=pbcDistance(getPosition(i),getPosition(j));
      const double value=distance.modulo();
	  const double invvalue=1.0/value;
      dist_array[j] = value;
      der_dist_array[j] = invvalue*distance  ; // 
    }
    vector<float> dummy=dist_array;
    vector<size_t> argsort_vec(sort_indexes(dummy)); // argsort distance array to get k-nearest neighbors
	
    for(unsigned k=1; k<(kcut+1); k+=1) {
      unsigned idx = argsort_vec[k];
	  dist_mat[i][k-1] += dist_array[idx];
	  idxn_mat[i][k-1] += idx;
      dist_deriv_mat[i][k-1] += -1.0*der_dist_array[idx];
    }
  }
  
  if(!serial) {
	comm.Sum(dist_mat); 
	comm.Sum(idxn_mat);
    comm.Sum(dist_deriv_mat);
  }
  
  //declare variables
  vector<double> input1=dist_mat.getVector(); 
  vector<double> input2=idxn_mat.getVector(); 
  vector<double> input3(nat, 1);

  //convert to tensor
  torch::Tensor input_efeat = torch::tensor(input1).view(-1);
  input_efeat.set_requires_grad(true);
  torch::Tensor input_idxn = torch::tensor(input2, torch::dtype(torch::kInt64)).view(-1);
  input_idxn.set_requires_grad(false);
  torch::Tensor input_nfeat = torch::tensor(input3).view({-1,1});
  input_nfeat.set_requires_grad(true);
 
  //convert to Ivalue
  std::vector<torch::jit::IValue> input;
  input.push_back( input_nfeat );
  input.push_back( input_idxn );
  input.push_back( input_efeat );

  //calculate output
  torch::Tensor output = _model.forward( input ).toTensor();  
  //set CV values
  vector<float> cvs = tensor_to_vector (output);
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
	Value* value = getPntrToComponent(name_comp);
	value->set(cvs[j]);
  }
}

// Define a cpp function for np.argsort() -> return sorted idx
vector<size_t> GraphGenerator::sort_indexes(const vector<float> &v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
      [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;   
}
    
} 

}

