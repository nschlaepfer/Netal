// CPU-only fixture builder. The image is hashed in RAM, not retained on the Mac.
#include <cstddef>
#include "nr_sampler_edge_inputs.h"
#include <CommonCrypto/CommonDigest.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
static std::string Sha(const void* p,size_t n){unsigned char digest[32];CC_SHA256(p,CC_LONG(n),digest);char out[65];for(int i=0;i<32;++i)snprintf(out+2*i,3,"%02x",digest[i]);return {out,64};}
int main(int argc,char** argv){try{
  if(argc!=2)throw std::runtime_error("usage: fixture FRESH_DIRECTORY");
  std::filesystem::path dir=argv[1];if(std::filesystem::exists(dir))throw std::runtime_error("preserve fixture");std::filesystem::create_directories(dir);
  auto image=EdgeImage();auto queries=EdgeQueries();
  std::ofstream q(dir/"queries.bin",std::ios::binary);q.write(reinterpret_cast<const char*>(queries.data()),std::streamsize(queries.size()*8));q.close();
  std::ofstream f(dir/"fixture.json");f<<"{\"schema\":\"neuralstudio.nr.sampler-edge-inputs.v1\",\"width\":"<<EDGE_W<<",\"height\":"<<EDGE_H<<",\"input_sha256\":\""<<Sha(image.data(),image.size()*2)<<"\",\"queries_sha256\":\""<<Sha(queries.data(),queries.size()*8)<<"\",\"queries\":"<<queries.size()<<",\"cases\":[";
  for(size_t i=0;i<EdgeCases().size();++i){const auto& c=EdgeCases()[i];f<<(i?",":"")<<"{\"name\":\""<<c.name<<"\",\"taps\":[";for(int j=0;j<4;++j)f<<(j?",":"")<<c.taps[j];f<<"]}";}
  f<<"]}\n";f.close();if(!f||!q)throw std::runtime_error("fixture write failed");
  std::cout<<"CPU fixture: "<<EdgeCases().size()<<" cases, "<<queries.size()<<" queries; no GPU execution\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
