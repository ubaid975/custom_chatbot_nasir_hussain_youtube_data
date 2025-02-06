import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
from langchain.embeddings import TensorflowHubEmbeddings,HuggingFaceEmbeddings
import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


os.environ['LANGSMITH_TRACING']="true"
os.environ['LANGSMITH_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGSMITH_API_KEY']=os.getenv('langsmith_api')
os.environ['LANGSMITH_PROJECT']="chat_with_nasir_hussain"
# OPENAI_API_KEY="<your-openai-api-key>"

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm=ChatGroq(api_key=os.getenv('groq_api'))
vector_db = Chroma(persist_directory='./db', embedding_function=embeddings)
# Define a custom prompt template (Modify as needed)
prompt_template = PromptTemplate.from_template(
    "Use the following context to answer the question you have no answer tell me i dont know:\n\n{context}\n\nQuestion: {question} "
)
memory=ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

prompt=PromptTemplate.from_template(
        """You are an AI assistant that provides accurate and concise answers.
Use the following retrieved documents to answer the question. If you don't know, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

retriever=vector_db.as_retriever(
    search_type='similarity',
    search_kwargs={'k':4}
)


con=ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt':prompt})

# Use the new RetrievalQA structure
retriever=vector_db.as_retriever(search_kwargs={"k": 3})


def chat(query,history):
    try:
        response=con.run({'question':query,'chat_history':history})
        return str(response)
    except Exception as e:
        return e

app=gr.ChatInterface(chat,theme=gr.themes.Soft(),title='Chat With Nasir Hussain Only Python Query'
                     ,description='I have provide llm Nasir Husssain Youtube Playlist Data Playlist Link Here:https://youtube.com/playlist?list=PLuYWhEqu9a9A7s21UXlZ1yYNPk5ZLfhpH&si=qg8iuts2csW3P4bQ')
app.launch(share=True)
