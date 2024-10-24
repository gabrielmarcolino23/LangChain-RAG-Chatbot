import os
import streamlit as st
from utils.pdf_loader import carregar_pdfs
from utils.exel_loader import carregar_excels
from utils.text_processing import processar_texto
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente virtual especializado em ajudar usuários com dúvidas na plataforma Oppem. Sempre que possível, baseie suas respostas nas informações presentes no documento fornecido para garantir que as respostas sejam precisas e atualizadas.
Lembre-se seja sempre direto e objetivo em suas respostas, fornecendo instruções claras e concisas para ajudar o usuário a resolver seu problema.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
)
# Já dei commit!

def main():
    st.set_page_config(page_title="💬 Chat-oppem", page_icon="🤖")

    st.title("💬 Chat-oppem")
    st.caption("🚀 Pergunte para nossa IA especialista em Oppem")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Você:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Inicialize Embeddings
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            print("Embeddings inicializados com sucesso.")
        except ImportError as e:
            st.error(f"Erro ao importar OpenAIEmbeddings: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Erro ao inicializar OpenAIEmbeddings: {e}")
            st.stop()

        # Carregue ou crie o vetorstore
        if os.path.exists('vectorstore/faiss_index'):
            try:
                vetorstore = FAISS.load_local('vectorstore/faiss_index', embeddings, allow_dangerous_deserialization=True)
                print("Vetorstore FAISS carregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao carregar o vetorstore FAISS: {e}")
                st.stop()
        else:
            try:
                vetorstore = criar_vetorstore(embeddings)
                print("Vetorstore FAISS criado e salvo com sucesso.")
            except Exception as e:
                st.error(f"Erro ao criar o vetorstore FAISS: {e}")
                st.stop()

        # Inicialize a memória da conversação
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            print("Memória da conversação inicializada.")

        # Configure o ConversationalRetrievalChain
        try:
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0,
                    model_name="gpt-4o-mini",  
                    max_tokens=500,
                ),
                retriever=vetorstore.as_retriever(search_kwargs={"k": 1}),
                memory=st.session_state.memory,
                chain_type="stuff",
                combine_docs_chain_kwargs={
                    "prompt": prompt_template
                },
                verbose=True
            )
            print("ConversationalRetrievalChain configurado com sucesso.")
        except Exception as e:
            st.error(f"Erro ao configurar ConversationalRetrievalChain: {e}")
            st.stop()

        # Obtenha a resposta do LLM
        try:
            resposta = qa({"question": user_input})
            print("Resposta obtida do LLM com sucesso.")
        except Exception as e:
            st.error(f"Erro ao obter a resposta do LLM: {e}")
            st.stop()

        # Adicione a resposta à sessão e exiba
        st.session_state.messages.append({"role": "assistant", "content": resposta['answer']})
        st.chat_message("assistant").write(resposta['answer'])

def criar_vetorstore(embeddings):
    textos = carregar_excels('docs/')  
    print(f"Texto extraído: {len(textos)} caracteres")
    if not textos.strip():
        raise ValueError("Nenhum texto foi extraído das planilhas Excel.")
    
    chunks = processar_texto(textos)
    print(f"Número de chunks criados: {len(chunks)}")
    if not chunks:
        raise ValueError("Nenhum chunk foi criado a partir do texto extraído.")
    
    vetorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vetorstore.save_local('vectorstore/faiss_index')
    print("Vetorstore criado e salvo com sucesso.")
    return vetorstore

if __name__ == "__main__":
    main()
