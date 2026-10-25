// Standalone oneDNN Graph SDPA layout probe. Graph pattern follows llama.cpp's
// fattn-onednn.cpp (MIT). No model weights are read or changed.
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <vector>

using namespace dnnl;
using namespace dnnl::graph;
using dt = logical_tensor::data_type;
using dims = logical_tensor::dims;
using layout = logical_tensor::layout_type;

struct plan {
    compiled_partition cp;
    std::vector<logical_tensor> inputs;
    logical_tensor output;
    size_t iq, ik, iv, is, im;
};

static plan build(const engine &eng, int nd, int H, int Hkv, int Q, int S, int D) {
    dims q{1,Hkv,H/Hkv,Q,D}, kv{1,Hkv,1,S,D};
    dims score{1,Hkv,H/Hkv,Q,S}, scale{1,1,1,1,1}, mask{1,1,1,Q,S};
    if(nd==4) { for(auto *v:{&q,&kv,&score,&scale,&mask}) v->erase(v->begin()); }
    size_t id=0;
    auto a=logical_tensor(id++,dt::f16,q,layout::strided);
    auto k=logical_tensor(id++,dt::f16,kv,layout::strided);
    auto s=logical_tensor(id++,dt::f32,score,layout::strided);
    auto mm1=op(id++,op::kind::MatMul,"qk");
    mm1.set_attr<bool>(op::attr::transpose_b,true);
    mm1.add_inputs({a,k}); mm1.add_outputs({s});
    auto c=logical_tensor(id++,dt::f16,scale,layout::strided);
    auto scaled=logical_tensor(id++,dt::f32,score,layout::strided);
    auto div=op(id++,op::kind::Divide,"scale");
    div.add_inputs({s,c}); div.add_outputs({scaled});
    auto m=logical_tensor(id++,dt::f16,mask,layout::strided);
    auto masked=logical_tensor(id++,dt::f32,score,layout::strided);
    auto add=op(id++,op::kind::Add,"mask");
    add.add_inputs({scaled,m}); add.add_outputs({masked});
    auto probs=logical_tensor(id++,dt::f16,score,layout::strided);
    auto sm=op(id++,op::kind::SoftMax,"softmax");
    sm.set_attr<int64_t>(op::attr::axis,-1);
    sm.set_attr<std::string>(op::attr::mode,"inf_as_zero");
    sm.add_inputs({masked}); sm.add_outputs({probs});
    auto v=logical_tensor(id++,dt::f16,kv,layout::strided);
    auto out=logical_tensor(id++,dt::f16,q,layout::strided);
    auto mm2=op(id++,op::kind::MatMul,"pv");
    mm2.add_inputs({probs,v}); mm2.add_outputs({out});
    dnnl::graph::graph g(eng.get_kind());
    for(const auto &o:{mm1,div,add,sm,mm2}) g.add_op(o);
    g.finalize();
    auto ps=g.get_partitions();
    if(ps.size()!=1 || !ps[0].is_supported()) throw std::runtime_error("SDPA not fused");
    plan p;
    p.inputs=ps[0].get_input_ports();
    p.output=ps[0].get_output_ports()[0];
    p.cp=ps[0].compile(p.inputs,{p.output},eng);
    p.output=p.cp.query_logical_tensor(p.output.get_id());
    p.iq=a.get_id();p.ik=k.get_id();p.iv=v.get_id();p.is=c.get_id();p.im=m.get_id();
    return p;
}

int main() try {
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    auto eng=dnnl::sycl_interop::make_engine(q.get_device(),q.get_context());
    auto stream=dnnl::sycl_interop::make_stream(eng,q);
    std::printf("device=%s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("layout,q,kv,mean_ms,max_abs_diff,nonfinite\n");
    constexpr int H=24,Hkv=4,D=256;
    for(auto shape:std::vector<std::pair<int,int>>{{64,256},{8192,32768},{8192,98304}}) {
        int Q=shape.first,S=shape.second;
        size_t nq=(size_t)H*Q*D,nkv=(size_t)Hkv*S*D,nm=(size_t)Q*S;
        auto *a=sycl::malloc_device<sycl::half>(nq,q);
        auto *k=sycl::malloc_device<sycl::half>(nkv,q);
        auto *v=sycl::malloc_device<sycl::half>(nkv,q);
        auto *m=sycl::malloc_device<sycl::half>(nm,q);
        auto *c=sycl::malloc_device<sycl::half>(1,q);
        auto *o=sycl::malloc_device<sycl::half>(nq,q);
        q.parallel_for(sycl::range<1>(nq),[=](sycl::id<1> x){ a[x]=(sycl::half)((int)(x[0]%29)-14)/29.0f; });
        q.parallel_for(sycl::range<1>(nkv),[=](sycl::id<1> x){
            k[x]=(sycl::half)(((int)(x[0]%31)-15)/31.0f);
            v[x]=(sycl::half)(((int)(x[0]%37)-18)/37.0f);
        });
        q.parallel_for(sycl::range<1>(nm),[=](sycl::id<1> x){
            size_t qi=x[0]/S,ki=x[0]%S;
            m[x]=(sycl::half)(ki <= S-Q+qi ? 0.0f : -INFINITY);
        });
        q.single_task([=]{*c=(sycl::half)16.0f;});
        q.wait_and_throw();
        std::vector<sycl::half> ref,host(nq);
        for(int nd:{5,4}) {
            auto p=build(eng,nd,H,Hkv,Q,S,D);
            auto ptr=[&](size_t id)->void*{
                if(id==p.iq)return a;if(id==p.ik)return k;if(id==p.iv)return v;
                if(id==p.is)return c;if(id==p.im)return m;return nullptr;
            };
            std::vector<tensor> ins;
            for(auto lt:p.inputs) ins.emplace_back(lt,eng,ptr(lt.get_id()));
            tensor out(p.output,eng,o);
            p.cp.execute(stream,ins,{out}); stream.wait();
            auto t0=std::chrono::steady_clock::now();
            int reps=Q<100?3:5;
            for(int r=0;r<reps;r++){p.cp.execute(stream,ins,{out});stream.wait();}
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/reps;
            q.memcpy(host.data(),o,nq*sizeof(sycl::half)).wait();
            double err=0;size_t nonfinite=0;
            if(nd==5)ref=host;
            for(size_t i=0;i<nq;i++){
                float f=(float)host[i];
                if(!std::isfinite(f))nonfinite++;
                err=std::max(err,(double)std::abs(f-(float)ref[i]));
            }
            std::printf("%d,%d,%d,%.6f,%.9f,%zu\n",nd,Q,S,ms,err,nonfinite);std::fflush(stdout);
            if(nonfinite || err>0.002) return 3;
        }
        for(auto ptr:{a,k,v,m,c,o})sycl::free(ptr,q);
    }
    return 0;
} catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
