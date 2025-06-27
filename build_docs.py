import os, re, html

def parse_exports():
    ns = open('NAMESPACE').read()
    m = re.search(r'export\(([^)]*)\)', ns, re.S)
    if not m:
        return []
    entries = [x.strip() for x in m.group(1).split(',')]
    entries = [e for e in entries if e and not e.startswith('#')]
    return entries

def extract_examples(path):
    with open(path) as f:
        lines = f.read().splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('\\examples{'):
            start = i
            break
    if start is None:
        return ''
    out = []
    brace = 1
    for l in lines[start+1:]:
        brace += l.count('{') - l.count('}')
        l_strip = l.strip()
        if l_strip.startswith('\\dont') or l_strip.startswith('\\dontrun'):
            continue
        if l_strip != '}' and l_strip != '':
            out.append(l)
        if brace <= 0:
            break
    return '\n'.join(out)

def main():
    exports = parse_exports()
    classes = [
        'ssa', '1d.ssa', '2d.ssa', 'nd.ssa', 'toeplitz.ssa', 'mssa', 'cssa',
        'ossa', 'pssa', 'wossa', 'series.list', 'lrr', 'forecast',
        'ssa.gaps', 'iossa.result', 'wcor.matrix', 'fdimpars.1d', 'fdimpars.nd',
        'hmatr'
    ]
    os.makedirs('docs/_build', exist_ok=True)
    with open('docs/_build/index.html', 'w') as f:
        f.write('<html><head><title>Rssa Documentation</title></head><body>')
        f.write('<h1>Available Classes</h1><ul>')
        for c in classes:
            f.write(f'<li>{c}</li>')
        f.write('</ul>')
        f.write('<h1>Exported Functions</h1><ul>')
        for name in exports:
            f.write(f'<li>{name}</li>')
        f.write('</ul>')
        examples = {
            'Forecasting': extract_examples('man/forecast.Rd'),
            'Recurrent Forecasting': extract_examples('man/rforecast.Rd'),
            'Gap Filling': extract_examples('man/gapfill.Rd'),
            'Iterative Gap Filling': extract_examples('man/igapfill.Rd'),
        }
        for title, code in examples.items():
            f.write(f'<h2>{title}</h2><pre>{html.escape(code)}</pre>')
        f.write('</body></html>')

if __name__ == '__main__':
    main()
