"""Read GGUF metadata and tensor type counts without loading model weights."""
import argparse,collections,json,struct
from pathlib import Path


def inspect(path):
    with Path(path).open('rb') as f:
        def read(fmt):
            size=struct.calcsize('<'+fmt);raw=f.read(size)
            if len(raw)!=size:raise ValueError('Truncated GGUF')
            return struct.unpack('<'+fmt,raw)[0]
        def string(keep=True):
            n=read('Q')
            if not keep:f.seek(n,1);return None
            return f.read(n).decode('utf-8',errors='replace')
        fmts={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        def value(t,keep):
            if t in fmts:return read(fmts[t])
            if t==8:return string(keep)
            if t==9:
                et,n=read('I'),read('Q')
                if et in fmts:f.seek(struct.calcsize('<'+fmts[et])*n,1)
                else:
                    for _ in range(n):value(et,False)
                return {'array_type':et,'count':n}
            raise ValueError('Unknown metadata type '+str(t))
        if f.read(4)!=b'GGUF':raise ValueError('Not a GGUF file')
        version=read('I');nt,nkv=read('Q'),read('Q');metadata={}
        for _ in range(nkv):
            k=string();keep=not k.startswith('tokenizer.') and 'chat_template' not in k
            v=value(read('I'),keep)
            if keep:metadata[k]=v
        types=collections.Counter();params=collections.Counter();mtp=[]
        for _ in range(nt):
            name=string();nd=read('I');shape=[read('Q') for _ in range(nd)];t=read('I');read('Q');types[str(t)]+=1
            n=1
            for dim in shape:n*=dim
            params[str(t)]+=n
            if 'nextn' in name:mtp.append({'name':name,'shape':shape,'ggml_type':t})
        return {'gguf_version':version,'tensor_count':nt,'file_bytes':Path(path).stat().st_size,'metadata':metadata,'tensor_type_counts':dict(types),'parameter_counts_by_type':dict(params),'nextn_tensors':mtp}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();s=json.dumps(inspect(a.model),ensure_ascii=False,indent=2)
    if a.output:a.output.write_text(s+'\n',encoding='utf-8')
    else:print(s)
