import streamlit as st
from utils import chatbot, text
from streamlit_chat import message

def main():
    st.set_page_config(page_title="Zoppy Mind", page_icon="🧠", layout="wide")

    st.title("Zoppy-Mind 🧠")
    st.caption("Tire suas dúvidas sobre o Zoppy 💬")

    user_input = st.text_input("Digite sua pergunta:")

    if(user_input):
        response = chatbot.get_response(user_input)

        message(user_input, is_user=True)
        message(response, is_user=False)

        

    pdf = '../docs/qg_zoppy.pdf'

    all_files = text.process_files(pdf)
    chunks = text.create_text_chunks(all_files)
    vectorstore = chatbot.create_embeddings(chunks)

    conversation = chatbot.create_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()