#ifndef __MODEL_H_
#define __MODEL_H_
#include<iostream>
#include"smartmap.hpp"

namespace maxent{
class Model {
public:
  inline int num_labels(void) {
    return _num_labels;
  }
  
  inline int num_dicts(void) {
    return _num_dicts;
  }

  void save(std::ostream & ofs);

  bool load(std::istream & ifs);

  int retrieve(int tid, const char * key, bool create = true);
  int index(int tid, const char * key, int lid = 0);
  int num_features();
  int dim();
  void set_num_labels(int num_labels);
  void set_num_dicts(int num_labels);

private:

  void write_unit(std::ostream & out, unsigned int val) {
    out.write(reinterpret_cast<const char *>(&val), sizeof(unsigned int));
  }

public:
  double * _W;
  double * _W_EEP;
  double * _W_EP;

  ltp::utility::IndexableSmartMap labels;
  ltp::utility::SmartMap<int> * dicts;
private:
  int _offset;
  int _num_labels;
  int _num_dicts;
  int _dim;
};//end class Model
}//end namespace maxent
#endif
