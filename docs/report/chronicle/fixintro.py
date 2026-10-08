import sys
p = sys.argv[1]; t = open(p).read()
i = t.find(" LaTeX version: ")
if i >= 0:
    j = t.index("}.", i) + 2
    t = t[:i] + r" The Markdown source is \code{docs/PROJECT\_CHRONICLE.md}." + t[j:]
open(p, "w").write(t)
