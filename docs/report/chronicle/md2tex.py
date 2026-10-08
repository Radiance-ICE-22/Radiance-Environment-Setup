"""Convert docs/PROJECT_CHRONICLE.md (a restricted Markdown subset) into the LaTeX body of project_chronicle.tex.
Figures are injected after given headings from figs.tex (blocks separated by '%%FIG <heading prefix>')."""
import re
import sys

src, figs_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src, encoding="utf-8").read().splitlines()

figs = {}
cur = None
for l in open(figs_path, encoding="utf-8").read().splitlines():
    if l.startswith("%%FIG "):
        cur = l[6:].strip(); figs[cur] = []
    elif cur is not None:
        figs[cur].append(l)

UNI = {"→": r"$\rightarrow$", "←": r"$\leftarrow$", "≤": r"$\le$", "≥": r"$\ge$", "×": r"$\times$", "≈": r"$\approx$",
       "·": r"$\cdot$", "−": r"$-$", "τ": r"$\tau$", "°": r"\textdegree{}", "…": r"\dots{}", "✔": r"\checkmark{}",
       "±": r"$\pm$", "Δ": r"$\Delta$", "▸": r"$\triangleright$", "↔": r"$\leftrightarrow$", "€": "EUR"}


def esc(t):
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    t = t.replace(r"\textbackslash\{\}", r"\textbackslash{}")
    return t


def inline(t):
    codes = []

    def keep(m):
        codes.append(r"\code{" + esc(m.group(1)).replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}").replace("--", "-{}-") + "}")
        return "\x00%d\x00" % (len(codes) - 1)
    s = re.sub(r"`([^`]*)`", keep, t)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = s.replace('"', "''")
    s = re.sub(r"(^|[\s(\[])''", r"\1``", s)
    for a_, b_ in UNI.items():
        s = s.replace(a_, b_)
    s = re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], s)
    return s


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    n = len(head)
    lens = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(n)]
    tot = sum(lens)
    if tot > 90:
        w = [max(0.11 * tot, l) for l in lens]
        avail = 1.0 - 0.026 * (n - 1)
        w = [avail * x / sum(w) for x in w]
        spec = "".join(r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % x for x in w)
    else:
        spec = "l" * n
    o = [r"\begin{center}\small", r"\begin{tabular}{@{}" + spec + "@{}}", r"\toprule",
         " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\", r"\midrule"]
    for r in body:
        r = (r + [""] * n)[:n]
        o.append(" & ".join(inline(c) for c in r) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return o


out_l = []
i = 0
para = []


def flush():
    global para
    if para:
        out_l.append(inline(" ".join(para)))
        out_l.append("")
        para = []


while i < len(lines):
    l = lines[i]
    if l.startswith("# "):
        flush(); i += 1; continue
    if l.startswith("## ") or l.startswith("### "):
        flush()
        lvl = "section" if l.startswith("## ") else "subsection"
        text = l.split(" ", 1)[1]
        text = re.sub(r"^\d+\.\s+", "", text) if lvl == "subsection" else text
        text = re.sub(r"^Part [IVX]+ — ", "", text) if lvl == "section" else text
        out_l.append("\\%s{%s}" % (lvl, inline(text)))
        for k, v in figs.items():
            if l.split(" ", 1)[1].startswith(k):
                out_l += v
        i += 1; continue
    if l.startswith("|"):
        flush()
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i]); i += 1
        out_l += table(rows); continue
    m = re.match(r"^(\s*)(- |\d+\. )(.*)", l)
    if m:
        flush()
        env = "itemize" if m.group(2) == "- " else "enumerate"
        items = []
        while i < len(lines):
            m2 = re.match(r"^(- |\d+\. )(.*)", lines[i])
            if m2:
                items.append(m2.group(2)); i += 1
            elif lines[i].startswith("  ") and items:
                items[-1] += " " + lines[i].strip(); i += 1
            else:
                break
        out_l.append(r"\begin{%s}" % env)
        out_l += [r"  \item " + inline(t) for t in items]
        out_l.append(r"\end{%s}" % env)
        continue
    if not l.strip():
        flush(); i += 1; continue
    para.append(l.strip()); i += 1
flush()
open(out, "w", encoding="utf-8").write("\n".join(out_l) + "\n")
