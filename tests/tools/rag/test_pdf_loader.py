from pypdf import PdfWriter

from rag_lib.local.loader import load_file


def test_pdf_loader_extracts_a_document(tmp_path):
    path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as output:
        writer.write(output)

    document = load_file(path)

    assert document.mime == "application/pdf"
    assert document.meta["format"] == "pdf"
    assert document.meta["pages"] == 1
    assert "[Page 1: no extractable text]" in document.text
