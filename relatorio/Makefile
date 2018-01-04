# Makefile options
SHELL=/bin/bash
.ONESHELL:
.PHONY: all clean deletar_lixo

# Other variables
MAIN_FILE = main
TEX_FILES = $(shell find -name '*.tex')
BIB_FILES = $(shell find -name '*.bib')
IMAGE_FILES = $(shell find -name '*.png' -or -name '*.jpg')
SOURCE_FILES = $(TEX_FILES) $(BIB_FILES) $(IMAGE_FILES)

all: $(MAIN_FILE).pdf

$(MAIN_FILE).pdf: $(SOURCE_FILES)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN_FILE)
	bibtex $(MAIN_FILE)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN_FILE)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN_FILE)
	$(MAKE) deletar_lixo

deletar_lixo:
	rm -f *.bbl *.idx *.lot *.toc *.blg *.lof *.aux *.brf *.log \
		*.fls *.ilg *.ind *_latexmk *.synctex.*

clean:
	$(MAKE) deletar_lixo
	rm -f $(MAIN_FILE).pdf
