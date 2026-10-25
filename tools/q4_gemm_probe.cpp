// oneDNN Q4 weight-decompression probe, independent of model weights.
// The Q4 primitive attributes follow Torchit1/llama.cpp sycl-b580-combined (MIT).
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>
using namespace dnnl;
using dt=memory::data_type;
using tag=memory::format_tag;

int main() {
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    auto name=q.get_device().get_info<sycl::info::device::name>();
    std::printf("device=%s\n",name.c_str());
    const auto *version=dnnl_version();
    std::printf("onednn=%d.%d.%d\n",version->major,version->minor,version->patch);
    if(name.find("B60")==std::string::npos) return 2;
    auto eng=sycl_interop::make_engine(q.get_device(),q.get_context());
    auto stream=sycl_interop::make_stream(eng,q);
    std::printf("mode,n,m,k,mean_ms,tera_ops_per_second,max_abs_diff,nrmse,nonfinite\n");
    for(auto shape:std::vector<std::vector<int>>{{128,256,256},{4096,17408,5120},{8192,17408,5120},{8192,5120,17408}}) {
        int N=shape[0],M=shape[1],K=shape[2];
        size_t na=(size_t)N*K,nw=(size_t)M*K,nc=(size_t)N*M,ns=(size_t)(K/32)*M;
        auto a=sycl::malloc_device<sycl::half>(na,q);
        auto w=sycl::malloc_device<sycl::half>(nw,q);
        auto u=sycl::malloc_device<uint8_t>(nw/2,q);
        auto scales=sycl::malloc_device<sycl::half>(ns,q);
        auto c=sycl::malloc_device<float>(nc,q);
        auto a8=sycl::malloc_device<int8_t>(na,q);
        auto w8=sycl::malloc_device<int8_t>(nw,q);
        auto source_scale=sycl::malloc_device<float>(N,q);
        auto weight_scale=sycl::malloc_device<float>(M,q);
        q.parallel_for(sycl::range<1>(N),[=](sycl::id<1> i){source_scale[i]=1.0f/32;});
        q.parallel_for(sycl::range<1>(M),[=](sycl::id<1> i){weight_scale[i]=1.0f/64;});
        auto zp=sycl::malloc_device<uint8_t>(1,q);
        q.memset(zp,8,1);
        q.parallel_for(sycl::range<1>(na),[=](sycl::id<1> i){a8[i]=(int8_t)((int)(i[0]%31)-15);a[i]=(sycl::half)((float)a8[i]/32.0f);});
        q.parallel_for(sycl::range<1>(ns),[=](sycl::id<1> i){scales[i]=(sycl::half)((1+(i[0]%3))/64.0f);});
        q.parallel_for(sycl::range<1>(nw/2),[=](sycl::id<1> ix){
            size_t i=2*ix[0],row=i/K,col=i%K;
            int lo=(int)((i*7+3)%16),hi=(int)(((i+1)*7+3)%16);
            u[ix]=(uint8_t)(lo|(hi<<4));
            float d=(float)scales[(col/32)*M+row];
            w[i]=(sycl::half)((lo-8)*d);w[i+1]=(sycl::half)((hi-8)*d);
            w8[i]=(int8_t)((lo-8)*(int)(d*64));w8[i+1]=(int8_t)((hi-8)*(int)(d*64));
        });
        q.wait_and_throw();
        std::vector<float> reference(nc),host(nc);
        for(int mode=0;mode<8;mode++) {
            try {
                auto a_md= mode==0 ? memory::desc({1,M,K},dt::f16,memory::dims{(int64_t)M*K,K,1}) : memory::desc({N,K},mode>=3?dt::s8:dt::f16,tag::ab);
                auto w_md= mode==0 ? memory::desc({1,K,N},dt::f16,memory::dims{(int64_t)N*K,1,K}) : memory::desc({K,M},(mode==2||mode==3)?dt::u4:mode>=4?dt::s8:dt::f16,tag::ba);
                auto c_md= mode==0 ? memory::desc({1,M,N},dt::f32,memory::dims{(int64_t)M*N,1,M}) : memory::desc({N,M},dt::f32,tag::ab);
                primitive_attr attr;attr.set_scratchpad_mode(scratchpad_mode::user);
                attr.set_fpmath_mode(fpmath_mode::f16,mode==2||mode==3);
                if(mode==2||mode==3) {
                    attr.set_scales(DNNL_ARG_WEIGHTS,3,{32,1},dt::f16);
                    attr.set_zero_points(DNNL_ARG_WEIGHTS,0,{},dt::u8);
                    if(M<=1024)attr.set_deterministic(true);
                }
                if(mode>=3&&mode<=5)attr.set_scales_mask(DNNL_ARG_SRC,mode==5?1:0);
                if(mode>=4&&mode<=5)attr.set_scales_mask(DNNL_ARG_WEIGHTS,mode==5?2:0);
                if(mode==7) {
                    post_ops ops;
                    ops.append_binary(algorithm::binary_mul,memory::desc({N,1},dt::f32,tag::ab));
                    ops.append_binary(algorithm::binary_mul,memory::desc({1,M},dt::f32,tag::ab));
                    attr.set_post_ops(ops);
                }
                auto pd=matmul::primitive_desc(eng,a_md,w_md,c_md,attr);
                auto scratch=sycl::malloc_device<uint8_t>(std::max<size_t>(1,pd.scratchpad_desc().get_size()),q);
                std::unordered_map<int,memory> args{
                    {DNNL_ARG_SRC,memory(a_md,eng,mode==0?(void*)w:mode>=3?(void*)a8:(void*)a)},
                    {DNNL_ARG_WEIGHTS,memory(w_md,eng,mode==0?(void*)a:(mode==2||mode==3)?(void*)u:mode>=4?(void*)w8:(void*)w)},
                    {DNNL_ARG_DST,memory(c_md,eng,c)},
                    {DNNL_ARG_SCRATCHPAD,memory(pd.scratchpad_desc(),eng,scratch)}};
                if(mode==2||mode==3) {
                    args.emplace(DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,memory(memory::desc({K/32,M},dt::f16,tag::ab),eng,scales));
                    args.emplace(DNNL_ARG_ATTR_ZERO_POINTS|DNNL_ARG_WEIGHTS,memory(memory::desc({1},dt::u8,tag::a),eng,zp));
                }
                if(mode>=3&&mode<=5)args.emplace(DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC,memory(memory::desc({mode==5?N:1},dt::f32,tag::a),eng,source_scale));
                if(mode>=4&&mode<=5)args.emplace(DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,memory(memory::desc({mode==5?M:1},dt::f32,tag::a),eng,weight_scale));
                std::printf("implementation,mode=%d,n=%d,m=%d,k=%d,%s\n",mode,N,M,K,pd.impl_info_str());
                if(mode==7) {
                    args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)|DNNL_ARG_SRC_1,memory(memory::desc({N,1},dt::f32,tag::ab),eng,source_scale));
                    args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(1)|DNNL_ARG_SRC_1,memory(memory::desc({1,M},dt::f32,tag::ab),eng,weight_scale));
                }
                matmul op(pd);
                auto execute=[&]() {
                    op.execute(stream,args);
                    if(mode==6)q.parallel_for(sycl::range<1>(nc),[=](sycl::id<1> idx){size_t i=idx[0];c[i]*=source_scale[i/M]*weight_scale[i%M];});
                    stream.wait();
                };
                execute();
                auto start=std::chrono::steady_clock::now();
                int repeats=N<1000?3:5;
                for(int i=0;i<repeats;i++)execute();
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
                q.memcpy(host.data(),c,nc*sizeof(float)).wait();
                if(mode==0) reference=host;
                double err=0,sse=0,ref2=0;size_t nonfinite=0;
                for(size_t i=0;i<nc;i++){
                    if(!std::isfinite(host[i]))nonfinite++;
                    double d=host[i]-reference[i];err=std::max(err,std::abs(d));sse+=d*d;ref2+=(double)reference[i]*reference[i];
                }
                double nrmse=std::sqrt(sse/std::max(ref2,1e-30));
                std::printf("%s,%d,%d,%d,%.6f,%.3f,%.8g,%.8g,%zu\n",mode==0?"f16_current":mode==1?"f16_swapped":mode==2?"q4_decompress":mode==3?"q4_w4a8":mode==4?"q8_w8a8":mode==5?"q8_axis_scales":mode==6?"q8_post_scale":"q8_fused_scales",N,M,K,ms,2.0*N*M*K/(ms*1e9),err,nrmse,nonfinite);std::fflush(stdout);
                sycl::free(scratch,q);
                if(nonfinite || nrmse>0.002)return 3;
            } catch(const dnnl::error &e) {
                std::printf("UNSUPPORTED mode=%d n=%d m=%d k=%d status=%d %s\n",mode,N,M,K,(int)e.status,e.what());std::fflush(stdout);
            }
        }
        for(auto ptr:{(void*)a,(void*)w,(void*)u,(void*)scales,(void*)c,(void*)zp,(void*)a8,(void*)w8,(void*)source_scale,(void*)weight_scale})sycl::free(ptr,q);
    }
}
