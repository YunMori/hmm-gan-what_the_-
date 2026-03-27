.PHONY: all clean view

all:
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view:
	open main.pdf
