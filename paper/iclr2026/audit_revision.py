"""Read-only integrity and prose checks for this manuscript revision.

Usage: python audit_revision.py [--framework-dir PATH]
The optional directory contains the unmodified upstream writing-framework
scripts. No research pipeline is executed and no files are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def rows(name):
    with (ROOT / 'results' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def uncomment(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', line) for line in text.splitlines())


def prose(text):
    """Project reader prose while excluding math, tables, and TeX plumbing.

    This is a manuscript-specific extraction, not a general TeX parser.
    The complete source and rendered pages are checked independently.
    """
    text = uncomment(text).split(r'\begin{iclrabstract}', 1)[1]
    for env in ('equation', 'tabular'):
        text = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', ' ', text, flags=re.S)
    text = re.sub(r'\$[^$]*\$', ' \u220e ', text, flags=re.S)
    text = re.sub(r'\\(?:cite[a-z]*|ref|label|includegraphics|bibliography|bibliographystyle|vspace|fontsize|begin|end)(?:\[[^]]*\])?\{[^}]*\}', ' ', text)
    text = re.sub(r'\\href\{[^}]*\}', '', text)
    text = re.sub(r'\\[A-Za-z]+\*?', '', text)
    text = text.replace(r'\%', '%').replace(r'\#', '#').replace('~', ' ')
    text = text.translate(str.maketrans({'{': '', '}': '', '[': '', ']': '', '\\': ' '}))
    return re.sub(r'[ \t]+', ' ', text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--framework-dir', type=Path)
    args = ap.parse_args()
    state = json.loads((HERE / 'manuscript_state.json').read_text(encoding='utf-8'))
    raw = (HERE / 'main.tex').read_bytes()
    text = raw.decode('utf-8').replace('\r\n', '\n')
    clean = uncomment(text)
    base = subprocess.check_output(['git', 'show', state['baseline_commit'] + ':paper/iclr2026/main.tex'], cwd=ROOT).decode('utf-8').replace('\r\n', '\n')
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})

    pattern = r'\\cite[a-z]*\*?(?:\[[^]]*\])*\{([^}]+)\}'
    keys = lambda s: {k.strip() for group in re.findall(pattern, s) for k in group.split(',')}
    bib = (HERE / 'references.bib').read_bytes()
    bibkeys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib.decode('utf-8')))
    check('existing citation-key set preserved', keys(text) == keys(base))
    check('every citation resolves', keys(text) <= bibkeys)
    check('bibliography byte-for-byte preserved', hashlib.sha256(bib).hexdigest() == state['bibliography_sha256'])
    check('all labelled equation bodies preserved', re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', text, re.S) == re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', base, re.S))
    labels = lambda s: sorted(re.findall(r'\\label\{([^}]+)\}', s))
    check('all labels preserved', labels(text) == labels(base))
    check('all cross-references resolve', set(re.findall(r'\\ref\{([^}]+)\}', text)) <= set(labels(text)))
    tables = lambda s: re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', s, re.S)
    numeric = lambda s: re.findall(r'[+-]?\d+(?:\.\d+)?', s)
    check('all numerical table content preserved', [numeric(t) for t in tables(text)] == [numeric(t) for t in tables(base)])
    comments = lambda s: [line[line.index('%'):] for line in s.splitlines() if re.match(r'^\s*%', line)]
    check('author comments preserved', comments(text) == comments(base))
    check('no rendered em-dash source forms', not re.search(r'---|\u2014|\\textemdash|\\emdash', clean))
    check('no unresolved editorial markers', not re.search(r'\[CLAIM NEEDS EVIDENCE\]|PLACEHOLDER_|\bTODO\b|\bXXX\b', clean))
    check('balanced braces', sum(1 for m in re.finditer(r'(?<!\\)\{',clean)) == sum(1 for m in re.finditer(r'(?<!\\)\}',clean)))

    paired = {r['arm']: r for r in rows('paired_tests_s.csv')}
    table = tables(text)[2]
    for arm, r in paired.items():
        for field, spec in [('mean_delta','+.6f'), ('ci_low','+.6f'), ('ci_high','+.6f'), ('t','+.2f'), ('cohen_dz','+.3f'), ('sd_paired','.6f')]:
            check(f'{arm} {field} matches archived summary', format(float(r[field]), spec) in table)
        check(f'{arm} p matches archived summary', format(float(r['p']), '.3f' if arm == 'random' else '.4f') in table)
        check(f'{arm} MDE matches archived SD and n', format(2.9 * float(r['sd_paired']) / math.sqrt(int(r['n_seeds'])), '.5f') in table)

    factorial = rows('factorial_s.csv')
    cells = {(int(r['seed']), r['arm']): r for r in factorial}
    seeds = {int(r['seed']) for r in factorial}
    check('24 complete cells at the reported token budget', len(cells) == 24 and len(seeds) == 8 and all(r['status'] == 'completed' and int(r['tokens_seen']) == 399900672 for r in factorial))
    for arm, r in paired.items():
        deltas = [float(cells[(seed, arm)]['final_val_loss']) - float(cells[(seed, 'baseline')]['final_val_loss']) for seed in sorted(seeds)]
        check(f'{arm} mean and sign agree with per-seed losses', math.isclose(sum(deltas)/len(deltas), float(r['mean_delta']), abs_tol=1e-12))

    for r in rows('reliability.csv'):
        for field in ('r_delta','r_stat','ceiling'):
            check(f"{r['model']} {field} matches Table 1", format(float(r[field]), '.3f') in tables(text)[0])
    for r in rows('a2_correlations.csv'):
        for field, spec in [('rho_raw','+.3f'),('ceiling','.3f'),('rho_disattenuated','+.3f')]:
            check(f"{r['model']} {r['statistic']} {field} matches Table 2", format(float(r[field]), spec) in tables(text)[1])

    from scipy.stats import spearmanr
    groups = defaultdict(list)
    for r in rows('a2_per_head.csv'):
        groups[r['model']].append(r)
    for r in rows('a2_correlations.csv'):
        source = groups[r['model']]
        stat = r['statistic']
        x = [(float(v[stat+'_half_a'])+float(v[stat+'_half_b']))/2 for v in source]
        y = [float(v['delta_pooled']) for v in source]
        check(f"{r['model']} {stat} pooled raw correlation reproduces", math.isclose(spearmanr(x,y).statistic, float(r['rho_raw']), abs_tol=1e-12))

    reader_text = prose(text)
    baseline_text = prose(base)
    result = {'status': 'PASS' if all(x['passed'] for x in checks) else 'FAIL', 'checks': checks,
              'source_sha256': hashlib.sha256(raw).hexdigest(),
              'pdf_sha256': hashlib.sha256((HERE/'out/main.pdf').read_bytes()).hexdigest(),
              'prose_projection_sha256': hashlib.sha256(reader_text.encode()).hexdigest(),
              'approximate_prose_words': {'baseline': len(baseline_text.split()), 'revised': len(reader_text.split())},
              'uncited_existing_bib_keys': sorted(bibkeys-keys(text))}
    if args.framework_dir:
        sys.path.insert(0, str(args.framework_dir.resolve()))
        from audit_candidate_text import audit_candidate
        from audit_text_consistency import audit_fragment
        result['raw_source_profile_findings'] = audit_fragment(clean, 'main.tex', state)
        result['candidate_prose_findings'] = audit_candidate(reader_text, 'main.tex prose projection', state)
        result['baseline_prose_findings'] = audit_candidate(baseline_text, 'baseline prose projection', state)
        framework_names = ('audit_candidate_text.py', 'audit_prose_patterns.py', 'audit_text_consistency.py', 'verify_paper.py')
        result['framework_script_sha256'] = {name: hashlib.sha256((args.framework_dir/name).read_bytes()).hexdigest() for name in framework_names}
        if result['raw_source_profile_findings'] or result['candidate_prose_findings']:
            result['status'] = 'FAIL'
    pdf = __import__('fitz').open(HERE/'out/main.pdf')
    rendered = '\n'.join(p.get_text() for p in pdf)
    result['pdf_pages'] = len(pdf)
    result['rendered_em_dash_count'] = rendered.count('\u2014')
    if result['rendered_em_dash_count']:
        result['status'] = 'FAIL'
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
