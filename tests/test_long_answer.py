import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from score_long_answer import score_answer


class AnswerTests(unittest.TestCase):
    def test_exact(self):
        result=score_answer('{"key":"ABC","count":67}',{"key":"ABC","count":67})
        self.assertTrue(result['all_correct'])
        self.assertTrue(result['format_exact_json'])

    def test_fenced(self):
        result=score_answer('```json\n{"key":"ABC"}\n```',{"key":"ABC"})
        self.assertTrue(result['all_correct'])
        self.assertFalse(result['format_exact_json'])

    def test_no_substrings_or_numeric_strings(self):
        result=score_answer('{"key":"ABC-extra","count":"67"}',{"key":"ABC","count":67})
        self.assertEqual(result['matched'],0)

    def test_first_answer_wins(self):
        result=score_answer('{"key":"wrong"} then {"key":"ABC"}',{"key":"ABC"})
        self.assertFalse(result['all_correct'])

    def test_missing_and_invalid(self):
        result=score_answer('invalid { key ABC',{"key":"ABC"})
        self.assertEqual(result['missing_keys'],['key'])


if __name__=='__main__':unittest.main()
