"""No weights or llama-cpp installation needed: python -m unittest -v test_decoding.
Checks algorithmic correctness, not model/backend integration or performance.
"""
import random
import unittest
from benchmark_common import decode

class Toy:
    def __init__(self, salt=0, constant=None):
        self.salt, self.constant = salt, constant
        self.prefix=[]
    def reset(self):
        self.prefix=[]
    def sync(self, ids):
        self.prefix=list(ids)
    def value(self):
        if self.constant is not None:
            return self.constant
        h=self.salt+7
        for t in self.prefix:
            h=(h*31+t+1)%1000003
        return h%11
    def next_token(self):
        return self.value()
    def block(self, ids):
        values=[]
        for t in ids:
            self.prefix.append(t)
            values.append(self.value())
        return values

class DecodingTests(unittest.TestCase):
    def test_matches_target_across_prompts_lengths_and_draft_errors(self):
        rng=random.Random(42)
        for _ in range(40):
            prompt=[rng.randrange(10) for _ in range(rng.randrange(1,12))]
            for limit in [1,2,5,17,32]:
                expected=decode(Toy(),prompt,{10},limit)['output_ids']
                for k in [1,2,3,8]:
                    for salt in [0,1,19]:
                        got=decode(Toy(),prompt,{10},limit,Toy(salt),k)
                        self.assertFalse(got['error'])
                        self.assertEqual(expected,got['output_ids'])
                        self.assertLessEqual(got['accepted_tokens'],got['proposed_tokens'])
    def test_immediate_end(self):
        result=decode(Toy(constant=10),[1],{10},10,Toy(constant=2),3)
        self.assertEqual(result['output_ids'],[])
        self.assertEqual(result['stop_reason'],'eog')
        self.assertIsNone(result['ttft_seconds'])
    def test_all_accepted_bonus_and_output_cap(self):
        result=decode(Toy(constant=1),[0],{10},7,Toy(constant=1),2)
        self.assertEqual(result['output_ids'],[1]*7)
        self.assertEqual(result['acceptance_rate'],1)
        self.assertEqual(len(result['events']),3)
    def test_wrong_draft_stop_is_corrected(self):
        result=decode(Toy(constant=1),[0],{10},4,Toy(constant=10),3)
        self.assertEqual(result['output_ids'],[1]*4)
        self.assertEqual(result['accepted_tokens'],0)
    def test_error_preserved(self):
        class Broken(Toy):
            def block(self, ids):
                raise RuntimeError('deliberate test failure')
        result=decode(Broken(),[1],{10},4,Toy(),2)
        self.assertEqual(result['stop_reason'],'error')
        self.assertIn('deliberate',result['error'])
    def test_soft_timeout(self):
        result=decode(Toy(),[1],{10},4,timeout=-1)
        self.assertEqual(result['stop_reason'],'timeout')

if __name__=='__main__':
    unittest.main()
