#include "input_cases.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,const char** argv) {
  try {
    if(argc!=2)throw std::runtime_error("usage: input_cases_probe NEW_OUTPUT");
    const std::filesystem::path output(argv[1]);
    if(std::filesystem::exists(output))throw std::runtime_error("preserve output");
    std::filesystem::create_directories(output);
    std::vector<uint16_t> pixels;
    for(unsigned variant=0;variant<4;++variant) {
      NrColorCase(pixels,1920,1080,variant);
      std::ofstream file(output/("case"+std::to_string(variant)+".rgba16le"),std::ios::binary);
      file.write(reinterpret_cast<const char*>(pixels.data()),static_cast<std::streamsize>(pixels.size()*sizeof(uint16_t)));
      if(!file)throw std::runtime_error("case write");
    }
    for(unsigned test=0;test<3;++test) {
      bool rejected=false;
      try {NrColorCase(pixels,test==0?0:16,test==1?0:16,test==2?4:0);}
      catch(const std::runtime_error&) {rejected=true;}
      if(!rejected)throw std::runtime_error("invalid case accepted");
    }
    std::cout<<"four color cases; invalid variant/width/height rejected\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
