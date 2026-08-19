from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def get_rag_chain():
    # 1. Load your vector store (assumes it's already populated and persisted)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Medical-specific system prompt safeguarding hallucinations
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI medical assistant specializing in cardiovascular and mental health support. "
         "Answer the user's question strictly using the provided context. If the answer cannot be found "
         "in the context, state that you do not know and advise consulting a healthcare professional. "
         "Never invent medical facts or prescriptions.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    # 3. Choose your LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 4. Construct the modern LCEL RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
