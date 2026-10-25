"""Adapt externally obtained Steve Miles Q4 MoE reorder to pinned b11869. See THIRD_PARTY_NOTICES.md."""
import argparse, hashlib, difflib, subprocess
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-dir', type=Path, required=True)
p.add_argument('--patch-file', type=Path, required=True)
p.add_argument('--output-patch', type=Path, required=True)
a = p.parse_args()
src = a.source_dir.resolve()
base = src / 'ggml' / 'src' / 'ggml-sycl'
if a.output_patch.exists():
    p.error('Output patch already exists')
commit = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
if commit != 'd327bf1f0780f87b8c01b367b61853b0c291d7c4':
    p.error('Expected clean b11869 source')
subprocess.run(['git', '-C', str(src), 'diff', '--exit-code', 'HEAD', '--', 'ggml/src/ggml-sycl/ggml-sycl.cpp', 'ggml/src/ggml-sycl/mmvq.cpp'], check=True, stdout=subprocess.DEVNULL)
raw = a.patch_file.read_bytes()
if hashlib.sha256(raw).hexdigest() != '5150ad8b8655ae80cd3f921a17605a5586fc8d976f43f3ce555ef92568885b87':
    p.error('Original patch SHA256 differs; inspect the upstream change before adapting')
patch = raw.decode('utf-8')
added = '\n'.join((line[1:] for line in patch.splitlines() if line.startswith('+') and (not line.startswith('+++'))))
start = added.index('static bool reorder_qw_q4_0_moe(')
end = added.index('// Reorder each Q8_0 expert slice')
func = added[start:end].rstrip() + '\n\n'
changes = []
pending = []
for filename in ['ggml-sycl.cpp', 'mmvq.cpp']:
    path = base / filename
    old = path.read_text(encoding='utf-8')
    new = old
    if filename == 'ggml-sycl.cpp':
        assert 'static bool reorder_qw_q4_0_moe(' not in old
        anchor = 'static bool reorder_qw_q8_0_moe('
        assert new.count(anchor) == 1
        new = new.replace(anchor, func + anchor, 1)
        anchor = 'switch (src0->type) {\n            case GGML_TYPE_Q8_0:\n                return reorder_qw_q8_0_moe'
        replacement = 'switch (src0->type) {\n            case GGML_TYPE_Q4_0:\n                return reorder_qw_q4_0_moe(data_device, src0->nb[2], src0->ne[2], stream);\n            case GGML_TYPE_Q8_0:\n                return reorder_qw_q8_0_moe'
        assert new.count(anchor) == 1
        new = new.replace(anchor, replacement, 1)
        anchor = 'if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&'
        assert new.count(anchor) == 1
        new = new.replace(anchor, 'if (src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&', 1)
    else:
        start = new.index('bool ggml_sycl_mul_mat_vec_q_id_reorder(')
        head = new[:start]
        b = new[start:]
        anchor = 'switch (src0_type) {\n        case GGML_TYPE_Q8_0:'
        replacement = 'switch (src0_type) {\n        case GGML_TYPE_Q4_0:\n            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(\n                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,\n                expert_weight_stride, dst_row_stride, src1_row_stride, stream);\n            return true;\n        case GGML_TYPE_Q8_0:'
        assert b.count(anchor) == 1
        b = b.replace(anchor, replacement, 1)
        new = head + b
    if filename == 'ggml-sycl.cpp':
        new = new.replace('// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.\nstatic bool reorder_qw_q4_0_moe', '// Reorder each Q4_0 expert slice into a self-contained [qs][d] SoA layout.\nstatic bool reorder_qw_q4_0_moe', 1)
        new = new.replace('static bool reorder_qw_q8_0_moe(', '// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.\nstatic bool reorder_qw_q8_0_moe(', 1)
        anchor = 'static void opt_for_reorder_id(ggml_backend_sycl_context * ctx, const ggml_tensor * src0) {\n'
        assert new.count(anchor) == 1
        new = new.replace(anchor, anchor + '    if (src0->type == GGML_TYPE_Q4_0) {\n        const char * value = std::getenv("GGML_SYCL_MOE_Q4_REORDER");\n        if (value && std::atoi(value) == 0) {\n            return;\n        }\n    }\n', 1)
    pending.append((path, old, new))
    changes.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/ggml/src/ggml-sycl/' + filename, tofile='b/ggml/src/ggml-sycl/' + filename))
    print(filename, hashlib.sha256(new.encode()).hexdigest())
for path, old, new in pending:
    path.write_text(new, encoding='utf-8')
a.output_patch.parent.mkdir(parents=True, exist_ok=True)
a.output_patch.write_text(''.join(changes), encoding='utf-8')
print('Applied Q4-only adaptation; keep the generated patch local and review upstream terms before redistribution.')
