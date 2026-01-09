import json
from tqdm import tqdm
from src.pdf_loader import load_documents
from src.indexer import build_vectorstore
from src.rag import build_chain
from src.answering import answer_question


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Building vector store...")
    db = build_vectorstore(documents)

    print("Building RAG chain...")
    chain = build_chain(db)

    with open("data/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    answers = []

    for q in tqdm(questions):
        value, refs = answer_question(chain, q["text"], q["kind"])
        answers.append({
            "question_text": q["text"],
            "value": value,
            "references": []
        })

    submission = {
        "team_email": "kosenko.kirill1000000@gmail.com",
        "submission_name": "kosenko_v4",
        "answers": answers
    }

    output_path = "submission_Kosenko_v4.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
