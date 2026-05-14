.PHONY: pdf figures clean

pdf:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

figures:
	python figures/generate_figures.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.toc
	rm -f figures/*.png