"""Compare large-only and small+large raw token IDs (no models required).
Usage: python compare_outputs.py results/LARGE_RUN results/SPECULATIVE_RUN
"""
import csv
import json
import sys
from pathlib import Path

def read(directory):
    directory=Path(directory)
    details={r['run_id']:r for r in (json.loads(s) for s in (directory/'details.jsonl').read_text(encoding='utf-8').splitlines())}
    with (directory/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return {(r['prompt_id'],r['repeat']):(r,details[r['run_id']]) for r in rows}

def main():
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python compare_outputs.py LARGE_RUN_DIRECTORY SPECULATIVE_RUN_DIRECTORY')
    left,right=map(read,sys.argv[1:])
    failed=0
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            print(key,'MISSING RUN')
            failed+=1
            continue
        a,ad=left[key]; b,bd=right[key]
        valid=(a['mode']=='large_only' and b['mode']=='small_large' and
               a['prompt']==b['prompt'] and a['stop_reason'] in ('eog','max_tokens') and
               b['stop_reason'] in ('eog','max_tokens'))
        match=valid and ad['output_ids']==bd['output_ids'] and a['stop_reason']==b['stop_reason']
        print(key,'MATCH' if match else 'MISMATCH / INVALID RUN')
        failed+=not match
    if not left or not right:
        failed+=1
    print('Confirm matching target model, template, output cap and hardware in manifests before interpreting this check.')
    raise SystemExit(1 if failed else 0)

if __name__=='__main__':
    main()
