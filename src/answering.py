import re
from decimal import Decimal, InvalidOperation


def answer_question(chain, question, kind):
    result = chain.invoke({"query": question})

    raw_answer = result["result"]
    value = postprocess(raw_answer, kind)

    references = []
    for doc in result["source_documents"]:
        references.append({
            "pdf_sha1": doc.metadata.get("pdf_sha1"),
            "page_index": int(doc.metadata.get("page_index", 0))
        })

    return value, references


def postprocess(raw, kind):
    text = raw.strip()

    if kind == "boolean":
        if text.lower().startswith("true") | text.lower().startswith("yes"):
            return True
        else:
            return False

    if kind == "number":
        return parse_number(text)

    if kind == "name":
        return text

    if kind == "names":
        return text

    return text


def parse_number(text: str):
    if not text:
        return "N/A"
    t = text.strip()
    t = t.replace("\u202f", " ")
    t = t.replace(" ", "")
    t = t.replace(",", "")
    match = re.search(r"-?\d+(\.\d+)?", t)
    if not match:
        return "N/A"
    try:
        num = Decimal(match.group())
    except InvalidOperation:
        return "N/A"
    if num == num.to_integral_value():
        return int(num)
    else:
        return float(num)
