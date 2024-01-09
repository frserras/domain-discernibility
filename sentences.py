import spacy
import polars as pl
from functools import reduce
from pathlib import Path
from spacy.tokens import DocBin


def set_extensions(docs):
    extensions = ["my_sents", "broad", "source_typology", "domain"]
    for doc in docs:
        for extension in extensions:
            if not doc.has_extension(extension):
                doc.set_extension(extension, default=None)
        yield doc


def tokens_to_sentences(tokens, starts):
    nexti = starts[1] if len(starts) > 1 else None
    sents = [tokens[starts[0] : nexti]]
    for end in starts[2:]:
        sents.append(tokens[nexti:end])
        nexti = end
    sents.append(tokens[nexti:])
    return sents


def load_docs(path, vocab):
    path = Path(path)
    docbin = DocBin()
    for f in path.iterdir():
        if f.is_file:
            docbin.merge(DocBin().from_disk(f))
    # docbin = reduce(lambda x, y: x.merge(y) or x,
    #                (DocBin().from_disk(f) for f in path.iterdir() if f.is_file))
    docs = docbin.get_docs(vocab)
    return docs


def keep_token(token):
    return token.is_alpha and not (
        token.is_space or token.is_punct or token.is_stop or token.like_num
    )


def keep_token_relaxed(token):
    return (token.is_alpha or token.is_punct) and not (
        token.is_space or token.like_num or token.is_quote
    )


def get_doc_lemmas(doc):
    return [t.lemma_ for t in doc if keep_token(t)]


def get_doc_tokens(doc):
    return [t.text.lower() for t in doc if keep_token(t)]


def get_doc_tokens_relaxed(doc):
    return [t.text.lower() for t in doc if keep_token_relaxed(t)]


def doc_to_dict(doc):
    # TODO: extract token.lemma_, token.text in one pass
    # tokens, lemmas, vectors = list(zip(*(((token.text, token.lemma_, token.vector) for token in doc))))
    # lemmas = list(zip(*(((token.text) for token in doc))))
    # sentence = list(map(get_doc_lemmas if LEMMATIZE else get_doc_tokens, tokens_to_sentences(list(doc), doc._.my_sents)))
    sentence = list(
        map(get_doc_tokens_relaxed, tokens_to_sentences(list(doc), doc._.my_sents))
    )
    d = {
        "broad": doc._.broad,
        "domain": doc._.domain,
        "source_typology": doc._.source_typology,
        "lemmatized_sentence": sentence,
        #'lemma'          : lemmas,
        #'token'          : tokens,
        #'vector'         : vectors,
    }
    return d


def dict_to_df(d):
    sent = pl.Series(
        name="sentence", values=[" ".join(s) for s in d["lemmatized_sentence"]]
    )
    broad = pl.Series(name="broad", values=[d["broad"] for s in sent])
    domain = pl.Series(name="domain", values=[d["domain"] for s in sent])
    source = pl.Series(
        name="source_typology", values=[d["source_typology"] for s in sent]
    )
    return pl.DataFrame([broad, domain, source, sent])


def doc_to_df(doc):
    return dict_to_df(doc_to_dict(doc))


def process(path, vocab):
    docs = load_docs(path, vocab)
    docs = set_extensions(docs)
    df = reduce(lambda x, y: x.vstack(y, in_place=True), map(doc_to_df, docs))
    df = df.filter(pl.col("sentence").str.lengths() > 2)
    return df


def create_categorical_labels(df):
    columns = ["broad", "domain", "source_typology"]
    for col in columns:
        d = {k[0]: v for v, k in enumerate(df.select(col).unique().iter_rows())}
        df.hstack(
            df.select(pl.col(col).map_dict(d).alias(col + "_label")), in_place=True
        )
    return df


def main():
    vocab = spacy.load("pt_core_news_lg").vocab
    df = process("./spacy", vocab)
    df = create_categorical_labels(df)
    df.write_csv("carolina_balanceado_sentenciado_rlx.csv")


if __name__ == "__main__":
    main()
