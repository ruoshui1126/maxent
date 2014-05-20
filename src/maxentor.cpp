#include "maxentor.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#define e 2.71828

namespace maxent {

Maxentor::Maxentor() {
};

Maxentor::~Maxentor() {
}

bool
Maxentor::read_instance(const char * train_file) {
  std::ifstream ifs(train_file);

  if(!ifs) {
    return false;
  }
  while(!ifs.eof()) {
    Instance * inst = new Instance;
    std::string line;
    std::getline(ifs,line);
    if(line.size() == 0) {
      delete inst;
      continue;
    }

    std::vector<std::string> words = split(line);
    if(words.size()<2) {
      std::cerr<<"instance ( "<<line<<" ) is illegal"<<std::endl;
      return false;
    }
    inst->label = words.back();
    words.pop_back();
    inst->features = words;
    for(int i = 0; i < inst->features.size(); ++i) {
      std::cout<<inst->features[i]<<",";
    }
    std::cout<<"label = "<<inst->label<<std::endl;
    train_dat.push_back(inst);
  }

  return true;
  
}

void
Maxentor::build_configration() {
  int dict_num = -1;  
  int offset = 0;
  for(int i = 0; i < train_dat.size(); ++i) {
    int size = train_dat[i]->features.size();
    if(size > dict_num) {
      dict_num = size;
    }
    train_dat[i]->label_id = model->labels.push(train_dat[i]->label);
  }

  model->set_num_dicts(dict_num); 
  model->set_num_labels(model->labels.size());

  model->dicts = new ltp::utility::SmartMap<int>[dict_num];

  ltp::utility::SmartMap<int>::const_iterator it = model->labels.begin();
  std::cout<<"#labels#"<<std::endl;
  while(it!=model->labels.end()) {
    const char * key = it.key() ;
    std::cout<<key<<std::endl;
    ++it;
  }
}

void
Maxentor::build_feature_space() {
  for(int i = 0; i < train_dat.size(); i++) {
    for(int j = 0; j< train_dat[i]->features.size(); j++ ) {
      model->retrieve(j, train_dat[i]->features[j].c_str());
      train_dat[i]->idx = model->index(j, train_dat[i]->features[j].c_str());
    }
  }
  std::cout<<"#feature space#"<<std::endl;
  
  for(int i = 0; i < model->num_dicts(); i++) {
    for(ltp::utility::SmartMap<int>::const_iterator it = model->dicts[i].begin(); it != model->dicts[i].end(); ++it) {
      std::cout<<it.key()<<" = "<<*it.value()<<std::endl;
    }
  }
  

  std::cout<<"dim = "<<model->dim()<<std::endl;
  model->_W = new double [model->dim()];
  model->_W_EEP = new double [model->dim()];
  model->_W_EP = new double [model->dim()];

  for(int i = 0; i< model->dim(); i++) {
    std::cout<<model->_W[i]<<",";
  }
  std::cout<<std::endl;
}

inline std::vector<std::string>
Maxentor::split(std::string str, int maxsplit ) {
    std::vector<std::string> ret;

  int numsplit = 0;
  int len = str.size();

  while (str.size() > 0) {
      size_t pos = std::string::npos;

      for (pos = 0; pos < str.size() && (str[pos] != ' '
                  && str[pos] != '\t'
                  && str[pos] != '\r'
                  && str[pos] != '\n'); ++ pos);

        if (pos == str.size()) {
          pos = std::string::npos;
        }

        if (maxsplit >= 0 && numsplit < maxsplit) {
            ret.push_back(str.substr(0, pos));
            ++ numsplit;
        } else if (maxsplit >= 0 && numsplit == maxsplit) {
            ret.push_back(str);
            ++ numsplit;
        } else if (maxsplit == -1) {
            ret.push_back(str.substr(0, pos));
            ++ numsplit;
        }

        if (pos == std::string::npos) {
            str = "";
        } else {
          for (; pos < str.size() && (str[pos] == ' '
                        || str[pos] == '\t'
                        || str[pos] == '\n'
                        || str[pos] == '\r'); ++ pos);
          str = str.substr(pos);
      }
  }

  return ret;
}//end split

void
Maxentor::calculate_eep() {
  for(int i = 0; i < train_dat.size(); i++ ) {
    Instance * inst = train_dat[i];
    for(int j = 0; j < inst->features.size(); j++) {
      std::cout<<inst->features[j]<<",";
      int id = model->index(j, inst->features[j].c_str());
      std::cout<<id<<",";
      model->_W_EEP[id+inst->label_id]+= 1.0;
    }
    std::cout<<inst->label <<","<<inst->label_id<<std::endl;
    
  }

  std::cout<<"#EEp#"<<std::endl;
  for(int i = 0; i < model->dim(); i++) {
    std::cout<<model->_W_EEP[i]<<",";
  }
  std::cout<<std::endl;
}

double 
Maxentor::newton(double eep, double ep, double param, double esp, double sigma2) {
  int max_iter=50;
  double x0=0.0, x=0.0;

  int N = train_dat.size();
  int C = model->num_dicts();
  for(int iter = 0; iter < max_iter; iter++) {
    double t= ep * pow(e,(model->num_dicts() * x0));
    double fval = t + N * (param + x0)/sigma2 - eep;
    double fpval = t * C + N / sigma2;

    if (fpval == 0) {
      std::cerr<<"WARNING: zero-division encounter in newtown() method"<<std::endl;
      return x0;
    }

    x = x0 - fval / fpval;
    if (x > x0){
      if(x - x0 < esp) {
        return x;
      }
    }

    else {
      if(x0 - x < esp) {
        return x;
      }
    }

    x0 = x;

  }
  std::cerr<<"ERROR: newtown method failed."<<std::endl;
}

void
Maxentor::calculate_ep() {
  int L = model->num_labels();

  for(int i = 0; i < model->dim(); i++) {
    model->_W_EP[i] = 0.0;
  }
  for(int i = 0; i < train_dat.size(); i++) {
    Instance * inst = train_dat[i];
    for(int j = 0; j < inst->features.size(); ++j) {
      int id = model->index(j, inst->features[j].c_str());
      double * temp = new double[L];
      double total = 0;
      for(int l = 0; l < L; l++) {
        temp[l] = model->_W[id + l]; 
        temp[l] = pow(e, temp[l]);
        total += temp[l];
      }

      for(int l = 0; l < L; l++) {
        temp[l] = temp[l] / total;
        model->_W_EP[id + l] += temp[l]; 
      }
    }
  }

  std::cout<<"#Ep#"<<std::endl;
  for(int i = 0; i < model->dim(); i++ ) {
    std::cout<<model->_W_EP[i]<<",";
  }
  std::cout<<std::endl;
}

void
Maxentor::train() {
  const char * train_file = "train.dat";
  if(!read_instance(train_file)) {
    std::cerr<<"Training file not exist."<<std::endl;
    return;
  }

  model = new Model();

  build_configration();

  build_feature_space();

  int max_iter = 10;

  calculate_eep();
  

  std::cout<<"e = "<<e<<std::endl;
  for(int i = 0; i < max_iter; i++) {
    std::cout<<"#iter# = [ "<<i<<"]"<<std::endl;
    calculate_ep();
    int param_size = model->dim();
    for(int j = 0; j < param_size; j++) {
      double inc = newton(model->_W_EEP[j], model->_W_EP[j], model->_W[j], 0.0001, 0.5); 
      model->_W[j] += inc;
    }

    std::cout<<"#W#"<<std::endl;
    for(int j = 0; j < param_size; j++) {
      std::cout<<model->_W[j]<<",";
    } 
    std::cout<<std::endl;
  }

  std::cout<<"#predict#"<<std::endl;

  std::vector<std::string> features;
  features.push_back("NOJB");
  features.push_back("low");
  predict(features);


}//end train

void
Maxentor::predict(std::vector<std::string> features) {
  int size = features.size();
  int L = model->num_labels();
  double * p = new double [L];
  for(int i = 0; i < size; i++) {
    int idx = model->index(i,features[i].c_str());
    for(int l = 0; l < L; l++) {
      p[l] += pow(e, model->_W[idx + l]); 
    } 
  }

  int max_index = -1;
  double max = -1;
  for(int l = 0; l < L; l++) {
    if(p[l] > max) {
      max = p[l];
      max_index = l;
    }
  }
}
}//end namespace
