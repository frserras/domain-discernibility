from pathlib import Path

import spacy
import polars as pl
from spacy.language import Language
from spacy.tokens import Doc, DocBin

_INPUTPATH = "./carolina_balanced_typologies.csv"
_OUTPUTDIR = "./spacy"
_EXTENSIONS = ["my_sents", "broad", "source_typology", "domain"]
_BATCHSIZE = 100


@Language.component("my_sents_component")
def my_sents_component(doc):
    starts = [i for i, t in enumerate(doc) if t.is_sent_start]
    doc._.my_sents = starts
    return doc


def setup():
    global _EXTENSIONS
    for extension in _EXTENSIONS:
        if not Doc.has_extension(extension):
            Doc.set_extension(extension, default=None)


def create_pipeline():
    nlp = spacy.load(
        "pt_core_news_lg", disable=["tok2vec", "parser", "ner", "attribute_ruler"]
    )
    nlp.enable_pipe("senter")
    nlp.add_pipe("my_sents_component", after="senter")
    nlp.max_length = 3_000_000
    return nlp


def outputfile(outputdir, i):
    return outputdir / "{:04d}.spacy".format(i)


def main():
    setup()
    nlp = create_pipeline()

    df = pl.read_csv(_INPUTPATH)
    texts = df["body"]
    labels = df.select(["domain", "source_type", "carolina_type"])

    outdir = Path(_OUTPUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    docBin = DocBin(store_user_data=True)

    for i, doc in enumerate(nlp.pipe(texts, n_process=6)):
        doc._.domain = labels["domain"][i]
        doc._.source_typology = labels["source_type"][i]
        doc._.broad = labels["carolina_type"][i]

        docBin.add(doc)
        if i % _BATCHSIZE == 0 and i > 0:
            docBin.to_disk(outputfile(outdir, i // _BATCHSIZE))
            docBin = DocBin(store_user_data=True)
    docBin.to_disk(outputfile(outdir, i // _BATCHSIZE + 1))


if __name__ == "__main__":
    main()
