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
   Contributing author: Connor (Oct2023)
   
------------------------------------------------------------------------- */
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "core/Atoms.h"
#include "tools/Tools.h"
#include "tools/Angle.h"
#include "tools/IFile.h"
#include "tools/SwitchingFunction.h"
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

//+PLUMEDOC COLVAR GRAPHMOL
/*
This file is used for biasing. It outputs the latent variable learnt by the ML model encoder. \
Several inputs need to be provided. CENTER corresponds to the position of nodes which \
is needed for neighborhood definitions and corresponding kNN algorithm. START1-4 are the starting atoms of \
intramolecular vector v_{1-4} and END1-4 are the ending atoms of v_{1-4}. KCUT is the number of nearest neighbor \
for kNN algorithm. MODEL takes a frozen pytorch model as input.

\plumedfile 
LOAD FILE=GraphMol.cpp

GRAPHMOL ...
 LABEL=model
 CENTER=COM
 START1=C
 END1=CA
 START2=N
 END2=C
 START3=N
 END3=CA
 START4=CHcenter
 END4=CA
 KCUT=6
 MODEL=model-local.pt
... GRAPHMOL

PRINT STRIDE=1  ARG=* FILE=COLVAR
\endplumedfile

*/
//+ENDPLUMEDOC

class MolecularGraphGenerator: 
  public Colvar
{
  bool serial;
  vector<AtomNumber> center_lista, start1_lista, end1_lista, start2_lista, end2_lista, start3_lista, end3_lista, start4_lista, end4_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  unsigned kcut, _n_out, _n_node, _n_edge, _n_edgefeat, _n_nodefeat, _n_atom;
  vector<size_t> sort_indexes(const vector<float> &v); // declare argsort function
  torch::jit::script::Module _model;
    
public:
  explicit MolecularGraphGenerator(const ActionOptions&);
  void calculate();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(MolecularGraphGenerator,"GRAPHMOL")

void MolecularGraphGenerator::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial");
  keys.add("atoms","CENTER","Reference particles for computation of neighborhood");
  keys.add("atoms","START1","Atom 1 for vector 1");
  keys.add("atoms","START2","Atom 1 for vector 2");
  keys.add("atoms","START3","Atom 1 for vector 3");
  keys.add("atoms","START4","Atom 1 for vector 4");
  keys.add("atoms","END1","Atom 2 for vector 1");
  keys.add("atoms","END2","Atom 2 for vector 2");
  keys.add("atoms","END3","Atom 2 for vector 3");
  keys.add("atoms","END4","Atom 2 for vector 4");
  keys.add("compulsory","KCUT","1","k-nearest neighbor for creating neighorlist"); // flag for k_cut in kNN
  keys.add("optional","MODEL","filename of the trained model"); 
  keys.addOutputComponent("node", "default", "NN outputs"); 

}

MolecularGraphGenerator::MolecularGraphGenerator(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  serial(false)
{  
  parseFlag("SERIAL",serial);
  parseAtomList("CENTER", center_lista); // list of node
  parseAtomList("START1", start1_lista); // list of vector1 atom1
  parseAtomList("END1", end1_lista); // list of vector1 atom2
  parseAtomList("START2", start2_lista); // list of vector2 atom1
  parseAtomList("END2", end2_lista); // list of vector2 atom2
  parseAtomList("START3", start3_lista); // list of vector2 atom1
  parseAtomList("END3", end3_lista); // list of vector2 atom2
  parseAtomList("START4", start4_lista); // list of vector2 atom1
  parseAtomList("END4", end4_lista); // list of vector2 atom2

  log.printf(" Parsing center list of size %f. \n", center_lista.size());
  log.printf(" Parsing start1 list of size %f. \n", start1_lista.size());
  log.printf(" Parsing end1 list of size %f. \n", end1_lista.size());
  log.printf(" Parsing start2 list of size %f. \n", start2_lista.size());
  log.printf(" Parsing end2 list of size %f. \n", end2_lista.size());
  log.printf(" Parsing start3 list of size %f. \n", start3_lista.size());
  log.printf(" Parsing end3 list of size %f. \n", end3_lista.size());
  log.printf(" Parsing start4 list of size %f. \n", start4_lista.size());
  log.printf(" Parsing end4 list of size %f. \n", end4_lista.size());

  _n_atom = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size()+end2_lista.size()+start3_lista.size()+end3_lista.size()+start4_lista.size()+end4_lista.size();
  
  parse("KCUT",kcut); // read knn parameter 
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

  _n_node = center_lista.size();

  _n_edge = center_lista.size() * kcut;
  _n_edgefeat = 4; // (4 intermolecular angles)
  _n_nodefeat = 1; // unity 
        
  //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<int> input_test1 ( _n_edge ); // edge index vector (1d tensor)
  std::vector<float> input_test2 ( _n_edge*_n_edgefeat ); // edge feature vector (2d tensor (N-node, D-feat))
  std::vector<float> input_test3 ( _n_node ); // node feature vector (2d tensor (N-node, D-feat))

  torch::Tensor single_input1 = torch::tensor(input_test1, torch::dtype(torch::kInt64));  
  torch::Tensor single_input2 = torch::tensor(input_test2).view({_n_edge, _n_edgefeat});
  torch::Tensor single_input3 = torch::tensor(input_test3).view({-1,_n_nodefeat});  
  
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
      
  atomsToRequest.reserve ( _n_atom  ); // Initilization of atoms list
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start1_lista.begin(), start1_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end1_lista.begin(), end1_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start2_lista.begin(), start2_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end2_lista.begin(), end2_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start3_lista.begin(), start3_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end3_lista.begin(), end3_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start4_lista.begin(), start4_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end4_lista.begin(), end4_lista.end() );
  requestAtoms(atomsToRequest);
}

// generator
void MolecularGraphGenerator::calculate()
{
  // clock_t begin_time = clock();
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
  
  Matrix<double> dist_mat(_n_node, kcut);
  Matrix<double> idxn_mat(_n_node, kcut);
  Matrix<double> theta1_mat(_n_node, kcut);
  Matrix<double> theta2_mat(_n_node, kcut);
  Matrix<double> theta3_mat(_n_node, kcut);
  Matrix<double> theta4_mat(_n_node, kcut);
  Matrix<Vector> dist_deriv_mat(_n_node, kcut);
  Matrix<Vector> theta1_deriv_mat(_n_node, 2*kcut);
  Matrix<Vector> theta2_deriv_mat(_n_node, 2*kcut);
  Matrix<Vector> theta3_deriv_mat(_n_node, 2*kcut);
  Matrix<Vector> theta4_deriv_mat(_n_node, 2*kcut);
  
  unsigned natom1 = center_lista.size();
  unsigned natom2 = center_lista.size()+start1_lista.size();
  unsigned natom3 = center_lista.size()+start1_lista.size()+end1_lista.size();
  unsigned natom4 = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size();
  unsigned natom5 = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size()+end2_lista.size();
  unsigned natom6 = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size()+end2_lista.size()+start3_lista.size();
  unsigned natom7 = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size()+end2_lista.size()+start3_lista.size()+end3_lista.size();
  unsigned natom8 = center_lista.size()+start1_lista.size()+end1_lista.size()+start2_lista.size()+end2_lista.size()+start3_lista.size()+end3_lista.size()+start4_lista.size();
  
  for(unsigned i=rank;i<_n_node;i+=stride) {
     
    vector<float> dist_array(_n_node);
	vector<Vector> distances(_n_node);
    vector<Vector> der_dist_array(_n_node);
	
	unsigned atom1_mol1=i+natom1;
	unsigned atom2_mol1=i+natom2;
	Vector v11=pbcDistance(getPosition(atom1_mol1), getPosition(atom2_mol1));
	
	unsigned atom3_mol1=i+natom3;
	unsigned atom4_mol1=i+natom4;
	Vector v12=pbcDistance(getPosition(atom3_mol1), getPosition(atom4_mol1));
	
	unsigned atom5_mol1=i+natom5;
	unsigned atom6_mol1=i+natom6;
	Vector v13=pbcDistance(getPosition(atom5_mol1), getPosition(atom6_mol1));
	
	unsigned atom7_mol1=i+natom7;
	unsigned atom8_mol1=i+natom8;
	Vector v14=pbcDistance(getPosition(atom7_mol1), getPosition(atom8_mol1));
      
    for(unsigned j=0;j<_n_node;j+=1) {
      Vector distance;
      distance=pbcDistance(getPosition(i),getPosition(j));
      const double value=distance.modulo();
	  const double invvalue=1.0/value;
      dist_array[j] = value;
      der_dist_array[j] = invvalue*distance ; 
	  distances[j] = distance; 
    }
    vector<float> dummy=dist_array;
    vector<size_t> argsort_vec(sort_indexes(dummy)); // argsort distance array to get k-nearest neighbors
	
    for(unsigned k=1; k<(kcut+1); k+=1) {
      unsigned idx = argsort_vec[k];
	  idxn_mat[i][k-1] += idx;
	  
	  unsigned atom1_mol2=idx+natom1;
	  unsigned atom2_mol2=idx+natom2;
	  Vector v21=pbcDistance(getPosition(atom1_mol2), getPosition(atom2_mol2));
	  
	  unsigned atom3_mol2=idx+natom3;
	  unsigned atom4_mol2=idx+natom4;
	  Vector v22=pbcDistance(getPosition(atom3_mol2), getPosition(atom4_mol2));
	  
	  unsigned atom5_mol2=idx+natom5;
	  unsigned atom6_mol2=idx+natom6;
	  Vector v23=pbcDistance(getPosition(atom5_mol2), getPosition(atom6_mol2));
	  
	  unsigned atom7_mol2=idx+natom7;
	  unsigned atom8_mol2=idx+natom8;
	  Vector v24=pbcDistance(getPosition(atom7_mol2), getPosition(atom8_mol2));
	  
	  Vector dv11, dv12, dv21, dv22, dv31, dv32, dv41, dv42;
	  PLMD::Angle a;
	  double theta1=a.compute(v11, v21, dv11, dv12);
	  double theta2=a.compute(v12, v22, dv21, dv22);
	  double theta3=a.compute(v13, v23, dv31, dv32);
	  double theta4=a.compute(v14, v24, dv41, dv42);
	  
	  // colvar;
	  theta1_mat[i][k-1] += theta1;
	  theta2_mat[i][k-1] += theta2;
	  theta3_mat[i][k-1] += theta3;
	  theta4_mat[i][k-1] += theta4;
	  // deriv; 
	  theta1_deriv_mat[i][k-1] -= dv11;
	  theta1_deriv_mat[i][k-1+kcut] -= dv12;
	  theta2_deriv_mat[i][k-1] -= dv21;
	  theta2_deriv_mat[i][k-1+kcut] -= dv22;
	  theta3_deriv_mat[i][k-1] -= dv31;
	  theta3_deriv_mat[i][k-1+kcut] -= dv32;
	  theta4_deriv_mat[i][k-1] -= dv41;
	  theta4_deriv_mat[i][k-1+kcut] -= dv42;
    }
  }
  
  if(!serial) {
	comm.Sum(idxn_mat);
	comm.Sum(theta1_mat);
	comm.Sum(theta2_mat);
	comm.Sum(theta3_mat);
	comm.Sum(theta4_mat);
	comm.Sum(theta1_deriv_mat);
	comm.Sum(theta2_deriv_mat);
	comm.Sum(theta3_deriv_mat);
	comm.Sum(theta4_deriv_mat);
  }

  
  //declare variables
  vector<double> input12=theta1_mat.getVector();
  vector<double> input13=theta2_mat.getVector();
  vector<double> input14=theta3_mat.getVector();
  vector<double> input15=theta4_mat.getVector();
  vector<double> input2=idxn_mat.getVector(); 
  vector<double> input3(_n_node, _n_nodefeat);

  //convert to tensor
  torch::Tensor theta1_feat = torch::tensor(input12).unsqueeze(/*dim=*/-1);
  torch::Tensor theta2_feat = torch::tensor(input13).unsqueeze(/*dim=*/-1);
  torch::Tensor theta3_feat = torch::tensor(input14).unsqueeze(/*dim=*/-1);
  torch::Tensor theta4_feat = torch::tensor(input15).unsqueeze(/*dim=*/-1);
  torch::Tensor input_efeat = torch::cat({theta1_feat,theta2_feat,theta3_feat,theta4_feat},/*dim=*/1); // concatenate all features 
  input_efeat.set_requires_grad(true);
  torch::Tensor input_idxn = torch::tensor(input2, torch::dtype(torch::kInt64)).view(-1);
  input_idxn.set_requires_grad(false);
  torch::Tensor input_nfeat = torch::tensor(input3).view({-1,_n_nodefeat});
  input_nfeat.set_requires_grad(false);

  //convert to Ivalue
  std::vector<torch::jit::IValue> input;
  input.push_back( input_nfeat );
  input.push_back( input_idxn );
  input.push_back( input_efeat );

  //calculate output
  torch::Tensor output = _model.forward( input ).toTensor();  
  //set CV values
  vector<float> cvs = tensor_to_vector (output);
  
  //derivatives
  for(unsigned j=0; j<_n_out; j++) {
    // expand dim to have shape (1,_n_out)
    int batch_size = 1;
    torch::Tensor grad_output = torch::ones({1}).expand({batch_size, 1});
    // calculate derivatives with automatic differentiation
	auto gradient = torch::autograd::grad({output.slice(/*dim=*/1, /*start=*/j, /*end=*/j+1)},
    /*outputing gradient will have the same shape to this input*/{input_efeat},
    /*grad_outputs=*/ {grad_output},
    /*retain_graph=*/true,
    /*create_graph=*/false);  // same shape to input_efeat;
    // add dimension
    torch::Tensor grad = gradient[0].unsqueeze(/*dim=*/1);
    //convert to vector
	vector<float> der = tensor_to_vector( grad );
    vector<Vector> deriv(_n_atom); 
	Tensor virial;
    //compute derivatives & virial
    for(unsigned i=rank;i<_n_node;i+=stride) {
	  for(unsigned k=0; k<kcut; k++) {
        unsigned idx = idxn_mat[i][k];
		unsigned ptr = i*kcut*_n_edgefeat+k*_n_edgefeat;
		
		// deriv
		deriv[i+natom1] += theta1_deriv_mat[i][k]*der[ptr];
		deriv[idx+natom1] += theta1_deriv_mat[i][k+kcut]*der[ptr];
		deriv[i+natom2] += -1.0*theta1_deriv_mat[i][k]*der[ptr];
		deriv[idx+natom2] += -1.0*theta1_deriv_mat[i][k+kcut]*der[ptr];
		deriv[i+natom3] += theta2_deriv_mat[i][k]*der[ptr+1];
		deriv[idx+natom3] += theta2_deriv_mat[i][k+kcut]*der[ptr+1];
		deriv[i+natom4] += -1.0*theta2_deriv_mat[i][k]*der[ptr+1];
		deriv[idx+natom4] += -1.0*theta2_deriv_mat[i][k+kcut]*der[ptr+1];
		deriv[i+natom5] += theta3_deriv_mat[i][k]*der[ptr+2];
		deriv[idx+natom5] += theta3_deriv_mat[i][k+kcut]*der[ptr+2];
		deriv[i+natom6] += -1.0*theta3_deriv_mat[i][k]*der[ptr+2];
		deriv[idx+natom6] += -1.0*theta3_deriv_mat[i][k+kcut]*der[ptr+2];
		deriv[i+natom7] += theta4_deriv_mat[i][k]*der[ptr+3];
		deriv[idx+natom7] += theta4_deriv_mat[i][k+kcut]*der[ptr+3];
		deriv[i+natom8] += -1.0*theta4_deriv_mat[i][k]*der[ptr+3];
		deriv[idx+natom8] += -1.0*theta4_deriv_mat[i][k+kcut]*der[ptr+3];

		// virial
		Vector v11 = pbcDistance(getPosition(i+natom1), getPosition(i+natom2));
		Vector v12 = pbcDistance(getPosition(idx+natom1), getPosition(idx+natom2));
		Vector v21 = pbcDistance(getPosition(i+natom3), getPosition(i+natom4));
		Vector v22 = pbcDistance(getPosition(idx+natom3), getPosition(idx+natom4));
		Vector v31 = pbcDistance(getPosition(i+natom5), getPosition(i+natom6));
		Vector v32 = pbcDistance(getPosition(idx+natom5), getPosition(idx+natom6));
		Vector v41 = pbcDistance(getPosition(i+natom7), getPosition(i+natom8));
		Vector v42 = pbcDistance(getPosition(idx+natom7), getPosition(idx+natom8));
	    Tensor th1_mol1(-1.0*theta1_deriv_mat[i][k]*der[ptr], v11);
	    Tensor th1_mol2(-1.0*theta1_deriv_mat[i][k+kcut]*der[ptr], v12);
	    Tensor th2_mol1(-1.0*theta2_deriv_mat[i][k]*der[ptr+1], v21);
	    Tensor th2_mol2(-1.0*theta2_deriv_mat[i][k+kcut]*der[ptr+1], v22);
		Tensor th3_mol1(-1.0*theta3_deriv_mat[i][k]*der[ptr+2], v31);
	    Tensor th3_mol2(-1.0*theta3_deriv_mat[i][k+kcut]*der[ptr+2], v32);
		Tensor th4_mol1(-1.0*theta4_deriv_mat[i][k]*der[ptr+3], v41);
	    Tensor th4_mol2(-1.0*theta4_deriv_mat[i][k+kcut]*der[ptr+3], v42);
		virial += th1_mol1+th1_mol2+th2_mol1+th2_mol2+th3_mol1+th3_mol2+th4_mol1+th4_mol2;
	  }
	}
	
	if(!serial) {
	  comm.Sum(virial);
	  comm.Sum(&deriv[0][0],3*getNumberOfAtoms());
	}
	
    //set derivatives of component j
    string name_comp = "node-"+std::to_string(j);
	Value* val=getPntrToComponent(name_comp);
    for(unsigned k=0; k<_n_atom; k++) {
      setAtomsDerivatives( val, k, deriv[k] );
    }
	setBoxDerivatives(val, virial);
	val->set(cvs[j]);
  }

}

// Define a cpp function for np.argsort() -> return sorted idx
vector<size_t> MolecularGraphGenerator::sort_indexes(const vector<float> &v) {
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

