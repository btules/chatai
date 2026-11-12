# =================================
# 1. IMPORTAÇÕES
# =================================
import os
import streamlit as st

# Injeta a chave como variável de ambiente
os.environ["HF_TOKEN"] = "INSERT_YOUR_HUGGINGFACE_TOKEN_HERE"
os.environ["GEMINI_API_KEY"] = "INSERT_YOUR_GEMINI_API_KEY_HERE"

# Loaders e chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings e LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# Vector Store
from langchain_community.vectorstores import Chroma
# Prompt
from langchain_core.prompts import PromptTemplate

# =================================
# 2. CONFIGURAÇÕES GERAIS
# =================================
PERSIST_DIRECTORY = "./chat_ai"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemini-3.5-flash"
USAR_RERANK = False   # mude para True depois que tudo funcionar


@st.cache_data
def carregar_documentos():
    caminhos = ["alura_whitePaper_devAprovaDeFuturo_2026.pdf"]
    documentos = []
    for caminho in caminhos:
        docs = PyPDFLoader(caminho).load()
        for doc in docs:
            doc.metadata["documento"] = caminho
        documentos.extend(docs)
    return documentos


def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documentos)


@st.cache_resource
def criar_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Se o banco já existe, só carrega
    if os.path.isdir(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

    chunks = gerar_chunks(carregar_documentos())
    if not chunks:
        st.error("Nenhum texto foi extraído do PDF (pode ser um PDF escaneado).")
        st.stop()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
        max_retries=5,
        timeout=60,
    )


def rerank_documentos(pergunta, documentos, llm):
    """Reranking opcional: uma nota de 0 a 10 por trecho (muitas chamadas!)."""
    import re

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template=(
            "Dê uma nota de 0 a 10 para a relevância do TRECHO para responder "
            "à PERGUNTA. Responda apenas com o número.\n\n"
            "PERGUNTA: {pergunta}\n\nTRECHO: {texto}\n\nNOTA:"
        ),
    )

    com_score = []
    for doc in documentos:
        resposta = llm.invoke(
            prompt_rerank.format(pergunta=pergunta, texto=doc.page_content)
        ).content
        achado = re.search(r"\d+(?:\.\d+)?", resposta)
        score = float(achado.group()) if achado else 0
        com_score.append((score, doc))

    com_score.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in com_score]


def responder_pergunta(pergunta, vectorstore):
    llm = get_llm()

    if USAR_RERANK:
        candidatos = vectorstore.similarity_search(pergunta, k=8)
        contexto_final = rerank_documentos(pergunta, candidatos, llm)[:4]
    else:
        contexto_final = vectorstore.similarity_search(pergunta, k=4)

    contexto_texto = "\n\n".join(doc.page_content for doc in contexto_final)

    prompt_final = f"""Você é um assistente que responde perguntas sobre o documento fornecido.
Use APENAS o contexto abaixo. Se a resposta não estiver no contexto, diga que
não encontrou essa informação no documento. Responda em português, de forma clara.

CONTEXTO:
{contexto_texto}

PERGUNTA:
{pergunta}

RESPOSTA:"""

    resposta = llm.invoke(prompt_final)
    return resposta.content, contexto_final


st.title("Chat AI - Dev do Futuro")

with st.spinner("Carregando modelos e documento..."):
    vectorstore = criar_vectorstore()

pergunta = st.text_input("Faça sua pergunta:")
if pergunta:
    with st.spinner("Pensando..."):
        resposta, fontes = responder_pergunta(pergunta, vectorstore)
    st.write(resposta)

    with st.expander("Trechos usados na resposta"):
        for doc in fontes:
            st.caption(f"{doc.metadata.get('documento')} — página {doc.metadata.get('page', 0) + 1}")
            st.write(doc.page_content)