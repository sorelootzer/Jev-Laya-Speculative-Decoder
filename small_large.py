"""Greedy speculative decoding: small drafts, large verifies in a block.
Includes first-mismatch correction, rejected-suffix rollback and bonus token.
Run python small_large.py --help.
"""
from benchmark_common import main
if __name__ == '__main__':
    main('small_large')
