#include "netal_nr_motion_geometry.h"
#include <cassert>
#include <cstdio>
#include <limits>
using namespace netal_nr_adapter;
static MotionGeometryInput observed() {
  MotionGeometryInput i;i.game_width=1920;i.game_height=1200;
  i.color_width=i.motion_width=1152;i.color_height=i.motion_height=720;
  i.motion_at_output_resolution=i.motion_includes_jitter=0;
  i.color_x=i.color_y=i.motion_x=i.motion_y=i.output_x=i.output_y=0;return i;
}
static void near(double a,double b) { assert(std::abs(a-b)<1e-9); }
int main() {
  MotionGeometry p;assert(PlanInputPixelMotion(observed(),p)==MotionGeometryError::None);
  assert(p.aspect.x==96 && p.aspect.y==0 && p.aspect.width==1728 && p.aspect.height==1080);
  near(p.vector_scale_x,1.5);near(p.vector_scale_y,1.5);
  ModelMotion v;assert(ConvertInputMotion(p,MotionRegion::Content,-10,4,v));near(v.x,-15);near(v.y,6);
  // Two explicit wrong hypotheses: game-output-only scaling and a sign flip.
  assert(std::abs(v.x-(-10.0*5.0/3.0))>1);assert(v.x!=15);
  auto first=MotionLocation(p,96,0),last=MotionLocation(p,1823,1079);
  assert(first.region==MotionRegion::Content && last.region==MotionRegion::Content);
  near(first.x,-1.0/6);near(first.y,-1.0/6);near(last.x,1151+1.0/6);near(last.y,719+1.0/6);
  assert(MotionLocation(p,95,0).region==MotionRegion::Padding);
  assert(MotionLocation(p,1824,1079).region==MotionRegion::Padding);
  assert(MotionLocation(p,1920,0).region==MotionRegion::Invalid);
  assert(MotionLocation(p,0,std::numeric_limits<uint32_t>::max()).region==MotionRegion::Invalid);
  unsigned content=0,padding=0;
  for(uint32_t y=0;y<1080;++y)for(uint32_t x=0;x<1920;++x){
    const auto q=MotionLocation(p,x,y);
    if(q.region==MotionRegion::Content){++content;
      // Pixel-center mapping round trip; padding translation applies to points,
      // NOT to vectors. This covers every admitted model pixel.
      near((q.x+.5)*p.aspect.width/p.input_width+p.aspect.x-.5,x);
      near((q.y+.5)*p.aspect.height/p.input_height+p.aspect.y-.5,y);
    }else {assert(q.region==MotionRegion::Padding);++padding;}
  }
  assert(content==1728*1080 && padding==192*1080);
  assert(ConvertInputMotion(p,MotionRegion::Padding,NAN,INFINITY,v));near(v.x,0);near(v.y,0);
  assert(!ConvertInputMotion(p,MotionRegion::Content,NAN,0,v));
  assert(!ConvertInputMotion(p,MotionRegion::Content,0,INFINITY,v));
  assert(!ConvertInputMotion(p,MotionRegion::Content,65504,0,v));near(v.x,0);
  assert(!ConvertInputMotion(p,MotionRegion::Invalid,0,0,v));
  for(unsigned k=0;k<13;++k){auto bad=observed();
    switch(k){case 0:bad.motion_at_output_resolution=-1;break;case 1:bad.motion_at_output_resolution=1;break;
      case 2:bad.motion_includes_jitter=-1;break;case 3:bad.motion_includes_jitter=1;break;
      case 4:bad.motion_x=1;break;case 5:bad.output_y=-1;break;case 6:bad.color_x=1;break;
      case 7:bad.motion_width=1920;break;case 8:bad.color_width=0;break;case 9:bad.scale_x=INFINITY;break;
      case 10:bad.scale_y=0;break;case 11:bad.game_height=1440;break;case 12:bad.color_height=16385;break;}
    assert(PlanInputPixelMotion(bad,p)!=MotionGeometryError::None);assert(!p.valid);
    assert(!ConvertInputMotion(p,MotionRegion::Content,1,1,v));
  }
  auto identity=observed();identity.game_height=1080;identity.color_width=identity.motion_width=1920;
  identity.color_height=identity.motion_height=1080;
  assert(PlanInputPixelMotion(identity,p)==MotionGeometryError::None);assert(p.aspect.identity());
  near(MotionLocation(p,100,200).x,100);near(MotionLocation(p,100,200).y,200);
  near(p.vector_scale_x,1);near(p.vector_scale_y,1);
  identity.scale_x=-2;identity.scale_y=.5;
  assert(PlanInputPixelMotion(identity,p)==MotionGeometryError::None);
  assert(ConvertInputMotion(p,MotionRegion::Content,3,-4,v));near(v.x,-6);near(v.y,-2);
  std::puts("PASS conditional motion geometry: all2073600 model centers, padding, scaling, sign counterexample,13 metadata refusals and finite/half-range guards. CPU only; no game-convention, interpolation or temporal-admission claim.");
}
