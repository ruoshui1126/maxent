#ifndef __MAXENTOR_H__
#define __MAXENTOR_H__

#include "model.h"
#include <iostream>
#include <string>
#include <vector>

namespace maxent {

class Instance{
  public:
    Instance(){}
    ~Instance(){}
  public:
    std::vector<std::string> features;
    int                      label_id;
    int                      idx;
    std::string              label;
};//end instance

class Maxentor {
public:
  Maxentor();
  ~Maxentor();
  void train();
  void predict(std::vector<std::string> features);

private:
  bool read_instance(const char * file_name);
  void build_configration(void);
  void build_feature_space(void);
  void calculate_eep(void);
  void calculate_ep(void);
  double newton(double eep, double ep, double param, double esp, double sigma2);
private:
  inline std::vector<std::string> split(std::string str,int maxsplit = -1);
public:

  std::vector<Instance *> train_dat;
  Model *                 model;
  
};//end Maxentor
}//end maxent
#endif
