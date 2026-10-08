# Rebuild docs/report/project_chronicle.{tex,pdf} from docs/PROJECT_CHRONICLE.md (figures: figs.tex, injected after headings).
S="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$S/../../.." && pwd)"; B=$(mktemp)
cd "$R"
python3 $S/md2tex.py docs/PROJECT_CHRONICLE.md $S/figs.tex $B || exit 1
python3 $S/fixintro.py $B
{ cat <<'PRE'
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,decorations.pathreplacing}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usepackage[hidelinks]{hyperref}
\setlist{itemsep=2pt,topsep=4pt}
\newcommand{\code}[1]{\texttt{#1}}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
\title{From FiGS to Semantic Goals:\\[2pt]
\large A Chronological Account of the Implementation, the Problems Met and How They Were Solved}
\author{Suhan --- Team Radiance, Department of Computer Science and Engineering,\\University of Moratuwa}
\date{8 October 2026}
\begin{document}
\maketitle
\begin{abstract}
This document records, in the order it happened, everything we implemented between July and October 2026: validating
and installing Stanford MSL's FiGS / SOUS-VIDE stack on our own hardware, a resumable capture-to-flight pipeline, the
Galley web console (job queue, splat page, course editor, SV-Net pages, splat editor), and the semantic layer that turns
a typed phrase into a 3D goal on a frozen Gaussian splat (teacher features, the lift, FMGS, the query, the evaluation
and four teacher designs). For each step it gives what was done, the problems met --- version drift, host faults,
silent data errors, GPU limits, power cuts, our own annotation mistakes --- and how each was diagnosed and fixed, with
the measured result.
\end{abstract}
\tableofcontents
\newpage
PRE
cat $B
echo '\end{document}'; } > docs/report/project_chronicle.tex
cd docs/report && for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error project_chronicle.tex > /dev/null 2>&1 || break; done
cd "$R/docs/report" && rm -f project_chronicle.{aux,log,out,toc} && echo "built docs/report/project_chronicle.pdf"
