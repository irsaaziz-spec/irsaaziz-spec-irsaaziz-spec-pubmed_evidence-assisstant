from pubmed_rag.build import to_abstract, write_index
from pubmed_rag.retrieval import Corpus


def test_to_abstract_labels_sections_and_leaves_out_the_conclusion():
    ctx = {"contexts": ["We studied X.", "Y improved."], "labels": ["BACKGROUND", "RESULTS"], "meshes": []}
    assert to_abstract(ctx) == "Background: We studied X.\nResults: Y improved."


def test_index_round_trip(tmp_path):
    docs = [{"doc_id": "1", "text": "alpha"}, {"doc_id": "2", "text": "beta"}]
    write_index(tmp_path, docs, [{"pubid": "1", "question": "q", "decision": "yes"}])
    c = Corpus.load(tmp_path)
    assert c.doc_ids == ["1", "2"] and c.texts == ["alpha", "beta"] and c.embeddings is None
