import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

PROMPT_TEMPLATE = """
You are a knowledgeable and friendly pharmacist providing clear, accurate, and professional guidance on medications based strictly on the provided context.


Context:
{context}

---
Question: {question}

---
Requirements:
1. Provide a direct, helpful, and easy-to-understand response as if speaking to a patient. Use **clear, natural language** and **structured bullet points** for key details.
2. Ensure the response is medically accurate, concise, and free from unnecessary technical jargon. If there are important precautions, interactions, or warnings, include them in a clear and approachable manner.
3. Also, do not use phrases like 'based on the context..' or something similar. 
4. If the context does not contain relevant information, state that explicitly instead of making assumptions.
5. Focus only on the necessary information without adding general reminders like “follow dosage instructions” unless the user specifically asks about it.
6. Do not include generic closing statements such as “consult your doctor if unsure” unless there are serious risks or interactions.
7. If the user should seek medical advice, mention it naturally within the response, rather than as a closing remark.
8. Only include storage instructions if refrigeration is required.
9. Avoid obvious warnings such as '"Do not use if the neck band or foil inner seal is missing'.
10. Make the word bold if the word directly answers users' question.

Keep the response engaging, medically sound, and free from unnecessary repetition.

"""



def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

# def main(query_text):
#     query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)
    
    model = ChatOpenAI(model_name="gpt-4", api_key="YOUR API KEY")
    response_text = model.invoke(prompt)


    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    # print(formatted_response)
    return response_text.content


# if __name__ == "__main__":
#     main()
