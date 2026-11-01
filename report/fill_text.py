"""Sentences that interpret the batch-size and GroupNorm(1) results.

Each %%KEY%% placeholder in make_report.py is replaced here; a block or list item whose
placeholder has no text yet (results still missing) is dropped.
"""


def texts(main, bs, ana, fa):
    t = {}
    return t


def fill(blocks, main, bs, ana, fa):
    t = texts(main, bs, ana, fa)

    def sub(s):
        for k, v in t.items():
            s = s.replace(f'%%{k}%%', v)
        return s

    out = []
    for b in blocks:
        b = dict(b)
        if 'text' in b:
            b['text'] = sub(b['text'])
            if '%%' in b['text']:
                continue
        if 'items' in b:
            b['items'] = [x for x in (sub(i) for i in b['items']) if '%%' not in x]
        out.append(b)
    return out
