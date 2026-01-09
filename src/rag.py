from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


def build_chain(db):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are a data extraction engine.
        
            Your task is to extract the correct answer to the question
            using ONLY the information provided in the context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            STRICT OUTPUT RULES:
            - Output MUST consist of exactly ONE line.
            - Do NOT include explanations, reasoning, or extra text.
            - Do NOT restate the question.
            - Do NOT use currency symbols or units.
            
            ANSWER FORMAT:
            - If the answer is a number:
              - Return ONLY the numeric value.
              - Convert words like thousand, million, billion, bn into numbers.
              - Example: "0.3 million" → 300000
            - If the answer is a percentage:
              - Return ONLY the numeric value (e.g. 56.4, not 56.4%).
            - If the answer is yes/no:
              - Return EXACTLY one of: True or False.
            - If the answer is a list:
              - Return a comma-separated list.
              - Do NOT include explanations or additional text.
            - If the answer is not explicitly stated in the context:
              - Return EXACTLY: N/A
            
            FINAL ANSWER:
        """
    )

    llm = Ollama(
        model="llama3",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa
