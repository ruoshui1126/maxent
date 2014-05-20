#include<map>
#include<iostream>
//using namespace std;

int main(){
  map<string,int> test;
  test["x=a"] = 10;
  test["x=b"] = 5;
  if(test.find("x=a")!=test.end()){
    std::cout<<test["x=a"]<<std::endl;
  }
  return 0;
}
