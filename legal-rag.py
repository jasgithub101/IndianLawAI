from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
import sounddevice as sd
import numpy as np
import wave
import tempfile
from langchain.schema import HumanMessage

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# Load Vectorstore
vectorstore = Chroma(
    persist_directory="law_books",
    embedding_function=embeddings,
)

retriever=vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve related sections or laws or acts",
    "You are a legal assistant specializing in Indian law. Your task is to retrieve the most relevant legal provisions, sections, and case laws based on the user’s query. Use the 'retriever_tool' only when the query requires specific legal references from Indian law, such as laws related to arrests, bail, evidence, or procedural matters. If the query is a general greeting like 'hi' or 'hello,' respond naturally without using any tool. Ensure retrieved information is precise, relevant, and directly applicable to the legal scenario described in the query.",    
)
tools=[retriever_tool]
retrieve=ToolNode([retriever_tool])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")

def AI_Assistant(state:AgentState):
    print("---CALL AGENT---")
    messages = state['messages']
    llm_with_tool = llm.bind_tools(tools)
        
    if len(messages) > 1:
        # llm.invoke({"messages": "Use the retrieved document only if the document contains keyword(s) or semantic meaning related to the user question. If the document is relevant, generate a summary or answer to the user question. If the document is not relevant, Answer the question based on your legal knowledge."})
        response = llm.invoke(messages[-1].content)
    else:
        response = llm_with_tool.invoke(messages)
    return {"messages": [response]}



def grade_documents(state:AgentState):
    llm_with_structure_op=llm.with_structured_output(grade)
    
    prompt=PromptTemplate(
         template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {documents} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"]
    )
    chain = prompt | llm_with_structure_op
    
    messages = state["messages"]
    last_message = messages[-1]
    
    question = messages[0].content
    docs = last_message.content
    
    scored_result = chain.invoke({"question": question, "documents": docs})
    print("retrieved doc")
    print(docs)
    score = scored_result.binary_score
    
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter"
    
def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")
    prompt += "If the retrieved document is still not relevant, answer the user question based on your legal knowledge."
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# Query Rewriting
def rewrite(state: AgentState):
    print("---TRANSFORM QUERY---")
    query = state["messages"][0].content
    message = [HumanMessage(content=f"Rewrite the following legal question for clarity: {query}")]
    response = llm.invoke(message)
    return {"messages": [response]}

workflow=StateGraph(AgentState)
workflow.add_node("ai_assistant",AI_Assistant)
workflow.add_node("retriever", retrieve) 
workflow.add_node("rewriter", rewrite) 
workflow.add_node("generator", generate)
workflow.add_edge(START,"ai_assistant")
workflow.add_conditional_edges("ai_assistant",tools_condition,
                               {"tools": "retriever",
                                END: END,})
workflow.add_conditional_edges("retriever",
                               grade_documents,
                               {"rewriter": "rewriter","generator": "generator"})
workflow.add_edge("generator", END)
workflow.add_edge("rewriter", "ai_assistant")

app=workflow.compile()

client = Groq()
def record_audio(duration=3, samplerate=44100):
    """Records 3 seconds of audio and saves it as a temp file."""
    st.info("Recording... Speak now!")

    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    return temp_wav.name

def audio_to_text(filepath):
    """Converts recorded audio to text using Whisper API."""
    with open(filepath, "rb") as file:
        translation = client.audio.translations.create(
            file=(filepath, file.read()),
            model="whisper-large-v3",
        )
    return translation.text

# Initialize Streamlit UI

st.title("AI for Indian Law")
st.write("")
if "query" not in st.session_state:
    st.session_state.query = ""

# User Input Field
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Enter your query:", value=st.session_state.query, key="query_input")  

with col2:
    if st.button("🎤"):
        audio_file = record_audio()
        recognized_text = audio_to_text(audio_file)
        st.session_state.query = recognized_text  # Update session state
        st.rerun()  # Force UI update

# Ensure session state updates with manual input
st.session_state.query = st.session_state.get("query_input", "")

if st.button("Search"):
    if st.session_state.query.strip():
        with st.spinner("Generating..."):
            response = app.invoke({"messages": [HumanMessage(content=st.session_state.query)]})
            st.subheader("Response:")
            st.write(response["messages"][-1].content)
    else:
        st.warning("Please enter a valid query.")
