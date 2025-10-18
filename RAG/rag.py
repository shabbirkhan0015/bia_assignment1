import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Environment Variable Setup ---
if not os.getenv("OPENAI_API_KEY"):
    st.sidebar.warning("OpenAI API Key not found. Please enter it below.", icon="⚠️")
    api_key_input = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
else:
    st.sidebar.success("OpenAI API Key is configured.", icon="✅")


def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text


def get_text_chunks(text):
    """Splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Creates FAISS vector store with OpenAI embeddings"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API Key is not set. Please enter it in the sidebar.")
        return None
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_conversational_chain():
    """Creates QA chain with custom prompt and OpenAI Chat model"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context".
    Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, vector_store):
    """Handles user questions with similarity search + QA chain"""
    if not user_question:
        st.warning("Please enter a question.")
        return

    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("### Answer:")
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"Error while processing question: {e}")


def main():
    st.set_page_config(page_title="PDF Question Answering (OpenAI)", page_icon="📄")
    st.header("Ask Questions about your PDF 💬")

    # Sidebar upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")

        if st.button("Submit & Process"):
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("Please provide the OpenAI API Key to proceed.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("Processing complete. You can now ask questions.")
                        else:
                            st.error("Failed to create vector store.")
                    else:
                        st.warning("No text extracted from PDFs.")

    # Main Q&A
    if "vector_store" in st.session_state:
        st.info("Your PDF has been processed. Ask a question below.")
        user_question = st.text_input("What would you like to know from the PDF?", key="user_question")
        if user_question:
            user_input(user_question, st.session_state.vector_store)
    else:
        st.info("Please upload a PDF and click 'Submit & Process' to start.")


if __name__ == "__main__":
    main()
