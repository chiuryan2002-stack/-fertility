#!/usr/bin/env python3
"""BPE / WordPiece / Unigram LM on zh-en-Amis parallel narratives.
Outputs results.json (consumed by index.html) and summary.csv.
No third-party libraries: the three algorithms are implemented here so that
training data, alphabet and budget are fully controlled."""
import re, json, math, bisect, csv, sys
from collections import Counter, defaultdict

UP = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
BUDGETS = [50, 150, 400, 1000]          # extra pieces beyond the character alphabet
ALGOS = ['BPE', 'WordPiece', 'Unigram LM']
SAMPLE = 100                            # lines kept per language for the explorer

# ---------------------------------------------------------------- RTF
def rtf_lines(path, enc):
    s = open(path, 'rb').read().decode('latin-1')
    s = s[s.index('\\f0\\fs24'):]
    assert '\\u' not in s
    lines, buf, j, n = [], bytearray(), 0, len(s)
    while j < n:
        c = s[j]
        if c == '\\':
            m = re.match(r"\\'([0-9a-fA-F]{2})", s[j:j + 4])
            if m: buf.append(int(m.group(1), 16)); j += 4; continue
            if j + 1 < n and s[j + 1] in '\r\n':
                lines.append(bytes(buf).decode(enc, 'replace')); buf = bytearray(); j += 2; continue
            m = re.match(r"\\[a-zA-Z]+-?\d* ?", s[j:])
            if m: j += m.end(); continue
            j += 2; continue
        if c in '{}\r\n': j += 1; continue
        buf.append(ord(c)); j += 1
    if buf: lines.append(bytes(buf).decode(enc, 'replace'))
    return [l for l in lines if l.strip()]

# ---------------------------------------------------------------- gold units
def amis_words(line):
    """'ma- sa- ma’an -tu' -> [[('ma','pre'),('sa','pre'),('ma’an','stem'),('tu','suf')]]"""
    words, pend = [], False
    for t in line.split():
        pre = t.endswith('-') and len(t) > 1
        suf = t.startswith('-') and len(t) > 1
        m = t.strip('-')
        if not m: continue
        role = 'inf' if (pre and suf) else 'pre' if pre else 'suf' if suf else 'stem'
        if (suf or pend) and words: words[-1].append((m, role))
        else: words.append([(m, role)])
        pend = pre
    return words

# English morphology by rule.  Inflection (-s/-es, -ed, -ing, clitics) is productive, so it is stripped by rule with the
# usual orthographic adjustments; derivation (-er, -ly, -ness, -ful, -less, -able, un-/re-/dis-) needs an attested base.
CLIT = ["n’t", "n't", "’s", "'s", "’re", "’ll", "’ve", "’d", "’m", "’"]
DER_SUF = ['ness', 'less', 'ful', 'able', 'ly', 'er']
DER_PRE = ['un', 're', 'dis']
NOSPLIT = set("""is was has his this us yes as always does its perhaps ours yours hers theirs besides sometimes towards
themselves wouls alas thus plus across something nothing anything everything morning evening during king spring string bring sing ring wing indeed
need speed seed bed red shed hundred recover thing""".split())
VOW = re.compile('[aeiouy]')
def en_inflect(w):
    """-> (stem, suffix) or None, following English spelling rules (e-drop, sibilant -es, consonant doubling)."""
    lw = w.lower()
    if lw in NOSPLIT or len(lw) < 4: return None
    if lw.endswith('ing') and len(lw) >= 5:
        st = w[:-3]
        if len(st) >= 2 and VOW.search(st.lower()):
            if len(st) >= 4 and st[-1] == st[-2] and st[-1].lower() not in 'lsfzaeiou' and st[-3].lower() in 'aeiou' and st[-4].lower() not in 'aeiou' \
               or (len(st) == 4 and st[-1] == st[-2] and st[-1].lower() not in 'lsfz' and st[-3].lower() in 'aeiou'):
                return w[:-4], w[-4:]                      # putting -> put|ting
            return st, 'ing'
    if lw.endswith('ed') and not lw.endswith('eed') and len(lw) >= 5:
        st = w[:-2]
        if len(st) >= 3 and VOW.search(st.lower()):
            if len(st) >= 4 and st[-1] == st[-2] and st[-1].lower() not in 'lsfz' and st[-3].lower() in 'aeiou':
                return w[:-3], w[-3:]                      # stopped -> stop|ped
            return st, 'ed'
    if lw.endswith('s') and not lw.endswith(('ss', 'us', 'is', "’s", "'s")):
        if lw.endswith('es') and len(lw) > 4:
            if re.search('(ses|ces|ges|zes)$', lw): return w[:-1], 's'        # house|s
            if not re.search('(s|x|z|ch|sh)es$', lw): return w[:-1], 's'      # time|s, tree|s
            return w[:-2], 'es'                                             # peach|es
        if len(lw) - 1 >= 3 and VOW.search(lw[:-1]): return w[:-1], 's'
    return None

def en_words(line, types):
    out = []
    for w in line.split():
        lw = w.lower(); done = None
        for c in CLIT:                                     # clitics / possessive first
            if lw.endswith(c) and len(lw) > len(c) and lw[:-len(c)].rstrip("’'").isalpha():
                k = len(w) - len(c); done = [(w[:k], 'stem'), (w[k:], 'suf')]; break
        if done and lw.endswith(('s’', "s'")) and done[1][0] in "’'" and w[:-1].lower().endswith('s') and w[0].islower():
            done = [(w[:-2], 'stem'), ('s', 'suf'), (w[-1], 'suf')]
        base = done[0][0] if done else w
        if not done or len(done) == 2:
            r = en_inflect(base) if base.isalpha() else None
            if r and base[0].isupper() and r[0].lower() not in types and r[0].lower() + 'e' not in types: r = None   # names
            if r and not (done and len(done) == 3):
                st, sf = r
                done = [(st, 'stem'), (sf, 'suf')] + ([d for d in done[1:]] if done else [])
        if not done:
            for suf in DER_SUF:
                st = lw[:-len(suf)]
                if lw.endswith(suf) and len(st) >= 4 and (st in types or st + 'e' in types) and lw not in NOSPLIT:
                    k = len(w) - len(suf); done = [(w[:k], 'stem'), (w[k:], 'suf')]; break
        if not done:
            for pre in DER_PRE:
                if lw.startswith(pre) and len(lw) - len(pre) >= 4 and lw[len(pre):] in types and lw not in NOSPLIT:
                    done = [(w[:len(pre)], 'pre'), (w[len(pre):], 'stem')]; break
        out.append(done or [(w, 'stem')])
    return out

def lab(m, role):
    return {'pre': m + '-', 'suf': '-' + m, 'inf': '-' + m + '-'}.get(role, m)

def build(name, raw):
    types = {re.sub(r"[’']", '', w.lower()) for l in raw for w in l.split()} if name == 'en' else None
    L = []
    for ln in raw:
        if name == 'zh':
            ws = ln.split()
            S = ''.join(ws); p = 0; ms = []
            for w in ws: ms.append((p, p + len(w), 'word', w)); p += len(w)
            L.append(dict(S=S, units=[(0, S)], ms=ms, ws=[(a, b, t) for a, b, _, t in ms], gold=' '.join(ws)))
        else:
            words = amis_words(ln) if name == 'ami' else en_words(ln, types)
            S = ''; units = []; ms = []; wsp = []; gold = []
            for wd in words:
                off = len(S); surf = ''.join(m for m, _ in wd); p = off
                for m, r in wd: ms.append((p, p + len(m), r, lab(m, r))); p += len(m)
                units.append((off, surf)); wsp.append((off, off + len(surf), surf))
                gold.append('-'.join(m for m, _ in wd)); S += surf
            L.append(dict(S=S, units=units, ms=ms, ws=wsp, gold=' '.join(gold)))
    return L

# ---------------------------------------------------------------- BPE
def train_bpe(units, M):
    words = dict(((tuple(u), c) for u, c in units.items())); merges = []
    for _ in range(M):
        pc = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]): pc[p] += c
        if not pc: break
        best, cnt = max(pc.items(), key=lambda x: (x[1], x[0]))
        if cnt < 2: break
        merges.append(best); words = merge_all(words, best, best[0] + best[1])
    return merges

def merge_all(words, pair, new):
    out = {}
    for w, c in words.items():
        if pair[0] in w:
            l, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]: l.append(new); i += 2
                else: l.append(w[i]); i += 1
            w = tuple(l)
        out[w] = out.get(w, 0) + c
    return out

def bpe_encode(u, rank, M):
    sy = list(u)
    while len(sy) > 1:
        best, bi = None, -1
        for i in range(len(sy) - 1):
            r = rank.get((sy[i], sy[i + 1]))
            if r is not None and r < M and (best is None or r < best): best, bi = r, i
        if best is None: break
        sy[bi:bi + 2] = [sy[bi] + sy[bi + 1]]
    return sy

# ---------------------------------------------------------------- WordPiece (likelihood-score merges, min pair freq 2)
def train_wp(units, M):
    words = {}
    for u, c in units.items():
        k = tuple([u[0]] + ['##' + ch for ch in u[1:]]); words[k] = words.get(k, 0) + c
    merges = []
    for _ in range(M):
        sc, pc = Counter(), Counter()
        for w, c in words.items():
            for s in w: sc[s] += c
            for p in zip(w, w[1:]): pc[p] += c
        cand = [(n / (sc[a] * sc[b]), n, (a, b)) for (a, b), n in pc.items() if n >= 2]
        if not cand: break
        _, _, (a, b) = max(cand)
        new = a + (b[2:] if b.startswith('##') else b)
        merges.append(new); words = merge_all(words, (a, b), new)
    return merges

def wp_vocab(units, merges, M):
    v = set(); [v.update([ch, '##' + ch]) for u in units for ch in u]
    v.update(merges[:M]); return v

def wp_encode(u, vocab):
    out, i = [], 0
    while i < len(u):
        j, cur = len(u), None
        while j > i:
            p = u[i:j] if i == 0 else '##' + u[i:j]
            if p in vocab: cur = j; break
            j -= 1
        if cur is None: return [u]
        out.append(u[i:cur]); i = cur
    return out

# ---------------------------------------------------------------- Unigram LM (EM + count-based pruning)
def lse(a, b):
    if a < b: a, b = b, a
    return a if b == -math.inf else a + math.log1p(math.exp(b - a))

def em(units, logp, ml, iters=2):
    cnt = {}
    for _ in range(iters):
        cnt = defaultdict(float)
        for u, c in units.items():
            n = len(u); a = [-math.inf] * (n + 1); a[0] = 0.0
            for i in range(n):
                if a[i] == -math.inf: continue
                for j in range(i + 1, min(n, i + ml) + 1):
                    lp = logp.get(u[i:j])
                    if lp is not None: a[j] = lse(a[j], a[i] + lp)
            b = [-math.inf] * (n + 1); b[n] = 0.0
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, min(n, i + ml) + 1):
                    lp = logp.get(u[i:j])
                    if lp is not None: b[i] = lse(b[i], lp + b[j])
            Z = a[n]
            for i in range(n):
                if a[i] == -math.inf: continue
                for j in range(i + 1, min(n, i + ml) + 1):
                    lp = logp.get(u[i:j])
                    if lp is not None: cnt[u[i:j]] += c * math.exp(a[i] + lp + b[j] - Z)
        tot = sum(cnt.values()); logp = {s: math.log(v / tot) for s, v in cnt.items() if v > 1e-9}
    return logp, cnt

def train_uni(units, Ms, ML=6):
    chars, sub = set(), Counter()
    for u, c in units.items():
        chars.update(u)
        for i in range(len(u)):
            for j in range(i + 1, min(len(u), i + ML) + 1): sub[u[i:j]] += c
    seed = {s: c for s, c in sub.items() if len(s) == 1 or c >= 2}
    tot = sum(seed.values()); logp = {s: math.log(c / tot) for s, c in seed.items()}
    res = {}
    for M in sorted(Ms, reverse=True):
        target = len(chars) + M
        while len(logp) > target:
            logp, cnt = em(units, logp, ML)
            for ch in chars: logp.setdefault(ch, -25.0)
            size = max(target, int(len(logp) * 0.75))
            multi = sorted((s for s in logp if len(s) > 1), key=lambda s: -cnt.get(s, 0))
            keep = set(multi[:max(0, size - len(chars))]) | chars
            logp = {s: v for s, v in logp.items() if s in keep}
        logp, _ = em(units, logp, ML)
        for ch in chars: logp.setdefault(ch, -25.0)
        res[M] = dict(logp)
    return res

def uni_encode(u, logp, ML=6):
    n = len(u); best = [-1e18] * (n + 1); bp = [0] * (n + 1); best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - ML), j):
            lp = logp.get(u[i:j])
            if lp is not None and best[i] + lp > best[j]: best[j], bp[j] = best[i] + lp, i
    out, j = [], n
    while j > 0: out.append(u[bp[j]:j]); j = bp[j]
    return out[::-1]

# ---------------------------------------------------------------- evaluation
def evaluate(lang, L, enc):
    ntok = nwords = nchars = 0; dist = [0] * 6; cls = [0] * 4
    TP = FP = FN = 0; brk = defaultdict(lambda: [0, 0, 0]); ex = defaultdict(Counter); tot = [0, 0, 0]
    stem = [0, 0, 0]; worst = defaultdict(lambda: [0, 0, Counter()]); cross = {}; ph = defaultdict(lambda: [0, 0])
    lines_out = []
    for li, ln in enumerate(L):
        S, n = ln['S'], len(ln['S']); pcs = []; B = set()
        for off, u in ln['units']:
            p = off
            for x in enc(u): pcs.append((p, p + len(x), x)); p += len(x); B.add(p)
        B.discard(n); Bs = sorted(B | {0, n}); Bset = set(Bs)
        Wb = {0, n}; [Wb.update([o, o + len(u)]) for o, u in ln['units']]
        G = set(); [G.update([a, b]) for a, b, _, _ in ln['ms']]; G |= Wb
        Gs = sorted(G); spanset = {(a, b) for a, b, _, _ in ln['ms']}
        Gi, Ti = G - Wb, B - Wb
        TP += len(Gi & Ti); FP += len(Ti - Gi); FN += len(Gi - Ti)
        ntok += len(pcs); nchars += n; nwords += len(ln['ws'])
        cs = []
        for s, e, x in pcs:
            inner = bisect.bisect_left(Gs, e) - bisect.bisect_right(Gs, s)
            if inner == 0: c = 0 if (s, e) in spanset else 1
            else: c = 2 if (s in G and e in G) else 3
            cls[c] += 1; cs.append(c)
            if lang == 'zh' and c >= 2:
                parts = []
                for a, b, t in ln['ws']:
                    if b <= s or a >= e: continue
                    seg = S[max(a, s):min(b, e)]; parts.append(seg + ('*' if (a < s or b > e) else ''))
                d = cross.setdefault(x, [0, '·'.join(parts)]); d[0] += 1
        for a, b, t in ln['ws']:
            k = 1 + bisect.bisect_left(Bs, b) - bisect.bisect_right(Bs, a)
            dist[min(k, 6) - 1] += 1
            w = worst[ln['gold'].split()[[w[0] for w in ln['ws']].index(a)] if lang != 'zh' else t]
            w[0] += 1; w[1] += k
            ins = [S[max(a, ps):min(b, pe)] for ps, pe, _ in pcs if ps < b and pe > a]
            w[2]['|'.join(ins)] += 1
        for a, b, r, lb in ln['ms']:
            if lang == 'zh' and b - a < 2: continue
            edge = a in Bset and b in Bset; inner = bisect.bisect_left(Bs, b) - bisect.bisect_right(Bs, a)
            st = 2 if not edge else (1 if inner else 0)
            if r == 'stem': stem[st] += 1; continue
            brk[lb][st] += 1; tot[st] += 1
            seg = '|'.join((('⟨' + x[:max(a, ps) - ps] + '⟩') if ps < a else '') + S[max(a, ps):min(b, pe)] +
                           (('⟨' + x[min(b, pe) - ps:] + '⟩') if pe > b else '')
                           for ps, pe, x in pcs if ps < b and pe > a)
            ex[lb][seg] += 1
        if lang == 'ami':
            spn = {(s, e) for s, e, _ in pcs}
            for m in re.finditer('ng', S):
                ph['ng'][0] += 1; ph['ng'][1] += (m.start() + 1) in B
            for i, ch in enumerate(S):
                if ch == '’': ph['glottal'][0] += 1; ph['glottal'][1] += (i, i + 1) in spn
        if li < SAMPLE:
            pp, k, cur = [], 0, []
            for off, u in ln['units']:
                cur = []
                while k < len(pcs) and pcs[k][0] < off + len(u): cur.append(pcs[k][2]); k += 1
                pp.append('|'.join(cur))
            lines_out.append(dict(g=ln['gold'], p=' '.join(pp), c=''.join(map(str, cs))))
    P = TP / (TP + FP) if TP + FP else 0; R = TP / (TP + FN) if TP + FN else 0
    F = 2 * P * R / (P + R) if P + R else 0
    rows = sorted(brk.items(), key=lambda kv: -sum(kv[1]))[:40]
    rows = [[k, sum(v), *v, ex[k].most_common(1)[0][0] if ex[k] else ''] for k, v in rows]
    wr = sorted(((k, v) for k, v in worst.items() if v[1] > v[0]), key=lambda kv: -(kv[1][1] - kv[1][0]))[:25]
    wr = [[k, v[0], round(v[1] / v[0], 2), v[2].most_common(1)[0][0]] for k, v in wr]
    cr = sorted(cross.items(), key=lambda kv: -kv[1][0])[:20]
    return dict(words=nwords, tokens=ntok, chars=nchars, fert=ntok / nwords, tpc=ntok / nchars, dist=dist, cls=cls,
                P=P, R=R, F=F, brk=rows, brk_tot=tot, stem=stem, worst=wr, cross=[[k, v[0], v[1]] for k, v in cr],
                ph={k: v for k, v in ph.items()}, lines=lines_out)

# ---------------------------------------------------------------- main
def main():
    raw = dict(ami=rtf_lines(UP + '前5篇_amis.rtf', 'cp1252'), en=rtf_lines(UP + '前5篇_english.rtf', 'cp1252'),
               zh=rtf_lines(UP + '前5篇_chinese.rtf', 'cp950'))
    C = {k: build(k, v) for k, v in raw.items()}
    U = {k: Counter(u for ln in C[k] for _, u in ln['units']) for k in C}
    U['joint'] = U['ami'] + U['en'] + U['zh']
    meta = dict(langs={}, budgets=BUDGETS, algos=ALGOS)
    for k in C:
        ms = [m for ln in C[k] for m in ln['ms']]
        meta['langs'][k] = dict(lines=len(C[k]), words=sum(len(l['ws']) for l in C[k]), chars=sum(len(l['S']) for l in C[k]),
                                types=len({t for l in C[k] for _, _, t in l['ws']}), morphs=len(ms),
                                affixes=sum(1 for m in ms if m[2] in ('pre', 'suf', 'inf')),
                                alphabet=len({c for l in C[k] for c in l['S']}))
    meta['alphabet_joint'] = len({c for k in C for l in C[k] for c in l['S']})
    Mx = max(BUDGETS); models = {}
    for key in ['joint', 'ami', 'en', 'zh']:
        b = train_bpe(U[key], Mx); w = train_wp(U[key], Mx)
        models[('BPE', key)] = ({p: i for i, p in enumerate(b)},); models[('WordPiece', key)] = (w, wp_vocab(U[key], w, 0))
        models[('Unigram LM', key)] = train_uni(U[key], BUDGETS)
        print(key, 'merges', len(b), len(w), file=sys.stderr)
    cfg = {}; cache = {}
    for algo in ALGOS:
        for mode in ['joint', 'mono']:
            for M in BUDGETS:
                res = {}
                for lang in C:
                    key = 'joint' if mode == 'joint' else lang
                    if algo == 'BPE': rank = models[(algo, key)][0]; f = lambda u: bpe_encode(u, rank, M)
                    elif algo == 'WordPiece':
                        merges = models[(algo, key)][0]; v = wp_vocab(U[key], merges, M); f = lambda u: wp_encode(u, v)
                    else: lp = models[(algo, key)][M]; f = lambda u: uni_encode(u, lp)
                    memo = {}
                    enc = lambda u, f=f, memo=memo: memo.setdefault(u, f(u))
                    res[lang] = evaluate(lang, C[lang], enc)
                cfg[f'{algo}|{mode}|{M}'] = res
                print(algo, mode, M, {k: round(v['fert'], 2) for k, v in res.items()}, file=sys.stderr)
    json.dump(dict(meta=meta, cfg=cfg), open(OUT + '/results.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    with open(OUT + '/summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['algorithm', 'training', 'extra_pieces', 'language', 'words', 'tokens', 'fertility',
                                        'tokens_per_char', 'boundary_P', 'boundary_R', 'boundary_F1',
                                        'tok_exact', 'tok_fragment', 'tok_merged', 'tok_straddle'])
        for k, res in cfg.items():
            a, m, M = k.split('|')
            for lang, r in res.items():
                w.writerow([a, m, M, lang, r['words'], r['tokens'], round(r['fert'], 3), round(r['tpc'], 3),
                            round(r['P'], 3), round(r['R'], 3), round(r['F'], 3), *r['cls']])

if __name__ == '__main__': main()
