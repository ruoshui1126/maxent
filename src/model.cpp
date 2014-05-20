#include "model.h"
namespace maxent{

int Model::num_features() {
  return _offset;
}

int Model::dim() {
  return _offset * _num_labels;
}
int Model::retrieve(int tid, const char * key, bool create) {
    int val;

    if (dicts[tid].get(key, val)) {
        return val;
    } else {
        if (create) {
            val = _offset;
            dicts[tid].set(key, val);
            ++ _offset;

            return val;
        }
    }
 
    return -1;
}

int Model::index(int tid, const char * key, int lid) {
    int idx = retrieve(tid, key, false);
    if (idx < 0) {
        return -1;
    }

    return idx * _num_labels + lid;
}

void Model::set_num_labels(int num_labels) {
  _num_labels = num_labels;
}

void Model::set_num_dicts(int num_dicts) {
  _num_dicts = num_dicts;
}

void Model::save(std::ostream & ofs) {
  char chunk1[16] = {'m','a','x','e','n','t','\0'}; 
  ofs.write(chunk1, 16);

  int off = ofs.tellp();

  unsigned labels_offset = 0;
  unsigned feature_offset = 0;
  unsigned parameter_offset = 0;

  write_unit(ofs, 0);
  write_unit(ofs, 0);
  write_unit(ofs, 0);

  labels_offset = ofs.tellp();
  labels.dump(ofs);

  feature_offset = ofs.tellp();
  char chunk2[16];
  unsigned sz = _num_dicts;
  strncpy(chunk2, "featurespace", 16);
  ofs.write(chunk2, 16);
  ofs.write(reinterpret_cast<const char *>(&_offset), sizeof(int));
  ofs.write(reinterpret_cast<const char *>(&sz), sizeof(unsigned int));

  for (int i = 0; i < _num_dicts; ++i) {
    dicts[i].dump(ofs);
  }

  parameter_offset = ofs.tellp();
  const double * p = _W;
  char chunk3[16] = {'p','a','r','a','m',0};
  ofs.write(chunk3,16);
  ofs.write(reinterpret_cast<const char *>(&_dim), sizeof(int));
  if(_dim > 0) {
    ofs.write(reinterpret_cast<const char *>(p), sizeof(double) * _dim);
  }

  ofs.seekp(off);
  write_unit(ofs, labels_offset);
  write_unit(ofs, feature_offset);
  write_unit(ofs, parameter_offset);
}
}//end maxent
