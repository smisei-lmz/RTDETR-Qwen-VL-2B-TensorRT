#include "edge.hpp"
#include <iostream>
using namespace edge;
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
int main(){try{
 Pool pool(1);auto first=*pool.acquire();require(!pool.acquire(),"pool exceeded capacity");pool.fill(first,synthetic(0,20,12));
 auto v=pool.read(first);require(color_detect(v).size()==1,"red box missing");pool.release(first);
 auto second=*pool.acquire();require(second.generation!=first.generation,"generation not advanced");
 bool rejected=false;try{pool.release(first);}catch(...){rejected=true;}require(rejected,"stale release accepted");pool.release(second);
 require(pool.all_free(),"lease leaked");
 Queue<int>q(2);require(!q.push(1),"unexpected drop");q.push(2);require(*q.push(3)==1,"must drop oldest");
 require(*q.pop()==2&&*q.pop()==3,"FIFO order");q.close();require(!q.pop(),"queue not empty");
 auto image=synthetic(0,20,12);auto tensor=preprocess_reference({image.rgb.data(),20,12},32,32);
 require(tensor.size()==32*32*3,"tensor shape");require(tensor[0]==0,"DETR must not pad input");
 Image corners{2,2,{0,0,0,255,0,0,0,255,0,0,0,255}};
 auto average=preprocess_reference({corners.rgb.data(),2,2},1,1);
 require(std::abs(average[0]-.25f)<1e-6&&std::abs(average[1]-.25f)<1e-6,"half-pixel RGB resize");
 float row[]={.1f,.2f,.8f,.9f,.9f,24.f};auto box=decode_detr_row(row,1000,500);
 require(box&&std::abs(box->x1-100)<.001&&std::abs(box->y2-450)<.001&&box->cls==24,"DETR normalized xyxy");
 row[4]=.1f;require(!decode_detr_row(row,1000,500),"DETR score threshold");
 row[5]=80;rejected=false;try{decode_detr_row(row,1000,500);}catch(...){rejected=true;}
 require(rejected,"invalid DETR class accepted");
 require(base64("abc")=="YWJj","base64 encoding");
 Settings s;s.lanes=1;auto cpu=make_cpu(s);Meta m;m.captured_mono=mono_ns();cpu->submit(0,{image.rgb.data(),20,12},m);
 require(cpu->available()==-1,"busy slot reused");require(cpu->poll().size()==1&&!cpu->busy(),"completion accounting");
 std::cout<<"PASS: pool generation, bounded queue, DETR resize/decode, base64, completion\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
