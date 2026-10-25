import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_unitrace import analyze, union_us


class TraceAnalysisTests(unittest.TestCase):
    def test_copy_wait_is_not_added_to_kernel_execution(self):
        # Request is [100, 200] us. Two overlapping kernels cover [110,170].
        # A synchronous copy command spans [100,180], including those kernels.
        events = [
            {'ph':'X','cat':'gpu_op','name':'gemm_kernel','ts':110,'dur':40,'pid':1,'tid':1},
            {'ph':'X','cat':'gpu_op','name':'micro_sdpa','ts':140,'dur':30,'pid':1,'tid':2},
            {'ph':'X','cat':'gpu_op','name':'zeCommandListAppendMemoryCopy(D2M)','ts':90,'dur':90,'pid':1,'tid':1},
            {'ph':'X','cat':'gpu_op','name':'gemm_kernel','ts':300,'dur':20,'pid':1,'tid':1},
            {'ph':'M','name':'process_name','ts':100,'args':{'name':'test'}},
        ]
        request = {'client_start_unix_ns':100000, 'client_end_unix_ns':200000}
        result, rows = analyze({'traceEvents':events}, request)
        self.assertAlmostEqual(result['kernel_union_seconds'], 60/1e6)
        self.assertAlmostEqual(result['memory_commands_outside_kernel_union_seconds'], 20/1e6)
        self.assertAlmostEqual(result['outside_kernel_union_seconds'], 40/1e6)
        self.assertEqual(result['kernel_union_percent'], 60)
        self.assertEqual(len(rows), 3)
        # Category sums can exceed their union when kernels run concurrently.
        total = sum(v['seconds'] for k,v in result['categories'].items() if k!='memory_command')
        self.assertAlmostEqual(total, 70/1e6)

    def test_union_handles_nested_disjoint_and_adjacent_ranges(self):
        self.assertEqual(union_us([]), 0)
        self.assertEqual(union_us([(10,20),(12,15),(20,25),(30,40)]), 25)


if __name__=='__main__':
    unittest.main()
