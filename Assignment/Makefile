TEX := $(wildcard *.tex)
PDF := $(TEX:.tex=.pdf)

.PHONY: all
all: $(PDF)

%.pdf: %.tex %.bbl
	@echo "========== PDF LaTeX (post BibTeX run) =========="
	pdflatex $<
	@echo "========== PDF LaTeX (final run) ================"
	pdflatex $<

%.bbl: %.aux
	@echo "========== BibTeX ==============================="
	bibtex $<

%.aux: %.tex
	@echo "========== PDF LaTeX (pre BibTeX run) ==========="
	pdflatex $<

.PHONY: clean
clean:
	rm -rf *.blg
	rm -rf *.log
	rm -rf *.out
