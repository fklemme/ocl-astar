TEX := $(wildcard *.tex)
PDF := $(TEX:.tex=.pdf)

.PHONY: all
all: AstarProgressAnimation/astar-0.png $(PDF)

AstarProgressAnimation/astar-%.png:
	mkdir -p AstarProgressAnimation
	convert -coalesce AstarProgressAnimation.gif AstarProgressAnimation/astar.png

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
	rm -rf AstarProgressAnimation
	rm -rf *.blg
	rm -rf *.log
	rm -rf *.nav
	rm -rf *.out
	rm -rf *.snm
	rm -rf *.toc
	rm -rf *.vrb
