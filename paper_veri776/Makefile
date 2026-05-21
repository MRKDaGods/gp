# paper_veri776/Makefile
# Requires: pdflatex, bibtex (or a full TeX distribution)
# Usage:
#   make           -> full build (pdflatex × 2, bibtex, pdflatex × 2)
#   make figures   -> regenerate figures from generate_figures.py
#   make clean     -> remove build artefacts

MAIN    = main
VENV    = ../.venv/Scripts/python.exe  # Windows venv; adjust for Linux

.PHONY: all figures clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib \
             tables/sota_comparison.tex tables/ablation.tex \
             figures/map_bar.pdf figures/pareto.pdf figures/ablation.pdf
	pdflatex -interaction=nonstopmode $(MAIN)
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

figures:
	$(VENV) figures/generate_figures.py

clean:
	-del /Q $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	         $(MAIN).out $(MAIN).pdf 2>nul
	# Linux: rm -f main.aux main.bbl main.blg main.log main.out main.pdf
