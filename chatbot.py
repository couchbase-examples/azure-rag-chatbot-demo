import streamlit as st
import os
import logging
import time
from datetime import timedelta

from dotenv import load_dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.exceptions import CouchbaseException
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_couchbase.cache import CouchbaseCache
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# --- Streamlit UI ---

st.set_page_config(page_title="Azure RAG Chatbot with Couchbase", layout="wide")
st.title("Azure RAG Chatbot with Couchbase")

# Initialize session state variables
if "config" not in st.session_state:
    st.session_state.config = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "cluster" not in st.session_state:
    st.session_state.cluster = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    st.subheader("Azure OpenAI")
    st.session_state.config["AZURE_OPENAI_KEY"] = st.text_input("Azure OpenAI Key", type="password", value=os.getenv("AZURE_OPENAI_KEY", ""))
    st.session_state.config["AZURE_OPENAI_ENDPOINT"] = st.text_input("Azure OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    st.session_state.config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = st.text_input("Embedding Deployment Name", value=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""))
    st.session_state.config["AZURE_OPENAI_CHAT_DEPLOYMENT"] = st.text_input("Chat Deployment Name", value=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""))
    st.session_state.config["AZURE_OPENAI_API_VERSION"] = st.text_input("API Version (e.g., 2024-02-01)", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))

    st.subheader("Couchbase")
    st.session_state.config["CB_HOST"] = st.text_input("Couchbase Host (e.g., couchbase://localhost)", value=os.getenv("CB_HOST", "couchbase://localhost"))
    st.session_state.config["CB_USERNAME"] = st.text_input("Couchbase Username", value=os.getenv("CB_USERNAME", "Administrator"))
    st.session_state.config["CB_PASSWORD"] = st.text_input("Couchbase Password", type="password", value=os.getenv("CB_PASSWORD", "password"))
    st.session_state.config["CB_BUCKET_NAME"] = st.text_input("Couchbase Bucket Name", value=os.getenv("CB_BUCKET_NAME", "vector-search-testing"))
    st.session_state.config["CB_SCOPE_NAME"] = st.text_input("Couchbase Scope Name", value=os.getenv("CB_SCOPE_NAME", "shared"))
    st.session_state.config["CB_COLLECTION_NAME"] = st.text_input("Couchbase Collection Name (for vectors)", value=os.getenv("CB_COLLECTION_NAME", "azure"))
    st.session_state.config["CB_INDEX_NAME"] = st.text_input("Couchbase Vector Index Name", value=os.getenv("CB_INDEX_NAME", "vector_search_azure"))
    st.session_state.config["CB_CACHE_COLLECTION"] = st.text_input("Couchbase Cache Collection Name", value=os.getenv("CB_CACHE_COLLECTION", "cache"))

    if st.button("Initialize Chatbot"):
        cfg = st.session_state.config
        missing_configs = [k for k, v in cfg.items() if not v]
        if missing_configs:
            st.error(f"Missing required configurations: {', '.join(missing_configs)}")
        else:
            with st.spinner("Initializing chatbot..."):
                try:
                    # Connect to Couchbase
                    auth = PasswordAuthenticator(cfg["CB_USERNAME"], cfg["CB_PASSWORD"])
                    options = ClusterOptions(auth)
                    cluster = Cluster(cfg["CB_HOST"], options)
                    cluster.wait_until_ready(timedelta(seconds=5))
                    st.session_state.cluster = cluster
                    logging.info("Successfully connected to Couchbase")

                    # Initialize Azure OpenAI Embeddings
                    embeddings = AzureOpenAIEmbeddings(
                        deployment=cfg["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                        openai_api_key=cfg["AZURE_OPENAI_KEY"],
                        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
                        api_version=cfg["AZURE_OPENAI_API_VERSION"]
                    )
                    logging.info("Successfully created AzureOpenAIEmbeddings")

                    # Initialize Couchbase Vector Store
                    vector_store = CouchbaseVectorStore(
                        cluster=cluster,
                        bucket_name=cfg["CB_BUCKET_NAME"],
                        scope_name=cfg["CB_SCOPE_NAME"],
                        collection_name=cfg["CB_COLLECTION_NAME"],
                        embedding=embeddings,
                        index_name=cfg["CB_INDEX_NAME"],
                    )
                    logging.info("Successfully created vector store")
                    
                    # Initialize Couchbase Cache
                    # Ensure the cache collection exists, but don't try to create it here
                    # as the notebook should have handled it.
                    # A check could be added to see if bucket/scope/collection exist if needed.
                    cache = CouchbaseCache(
                        cluster=cluster,
                        bucket_name=cfg["CB_BUCKET_NAME"],
                        scope_name=cfg["CB_SCOPE_NAME"],
                        collection_name=cfg["CB_CACHE_COLLECTION"],
                    )
                    set_llm_cache(cache)
                    logging.info("Successfully created and set LLM cache")

                    # Initialize Azure OpenAI Chat Model
                    llm = AzureChatOpenAI(
                        deployment_name=cfg["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                        openai_api_key=cfg["AZURE_OPENAI_KEY"],
                        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
                        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"]
                    )
                    logging.info("Successfully created Azure OpenAI Chat model")

                    # Create RAG chain
                    template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
                    {context}
                    Question: {question}"""
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    st.session_state.rag_chain = (
                        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    logging.info("Successfully created RAG chain")
                    st.session_state.initialized = True
                    st.success("Chatbot initialized successfully!")
                    st.session_state.messages = [] # Clear messages on re-init
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
                    logging.error(f"Initialization failed: {str(e)}")
                    st.session_state.initialized = False
                    if st.session_state.cluster:
                        try:
                            st.session_state.cluster.disconnect()
                            logging.info("Disconnected from Couchbase due to initialization error.")
                        except Exception as disconnect_e:
                            logging.error(f"Error disconnecting from Couchbase: {disconnect_e}")
                        st.session_state.cluster = None


# Display chat messages from history on app rerun
if st.session_state.initialized:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("Please configure and initialize the chatbot using the sidebar.")

# Accept user input
if prompt := st.chat_input("What is your question?"):
    if not st.session_state.initialized:
        st.warning("Please initialize the chatbot first using the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                start_time = time.time()
                response = st.session_state.rag_chain.invoke(prompt)
                end_time = time.time()
                
                full_response = response
                message_placeholder.markdown(full_response)
                
                # Log response time
                logging.info(f"RAG response generated in {end_time - start_time:.2f} seconds for query: {prompt}")
                
            except CouchbaseException as e:
                full_response = f"Couchbase error: {str(e)}"
                message_placeholder.error(full_response)
                logging.error(f"Couchbase error during RAG invoke: {str(e)}")
            except Exception as e:
                full_response = f"An error occurred: {str(e)}"
                message_placeholder.error(full_response)
                logging.error(f"Error during RAG invoke: {str(e)}")
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a disconnect button for cleanup if needed, or handle in on_stop if Streamlit supports it.
# For now, relying on Streamlit's session management.
# If running in an environment where explicit disconnect is critical,
# you might need to manage the cluster object more carefully.
# Upon closing the app/session, cluster connections should ideally be closed.
# Streamlit's execution model might make explicit cleanup tricky without specific hooks.

def on_stop():
    if st.session_state.get("cluster"):
        try:
            st.session_state.cluster.disconnect()
            logging.info("Disconnected from Couchbase on app stop.")
        except Exception as e:
            logging.error(f"Error disconnecting from Couchbase on app stop: {e}")

# Note: Streamlit does not have a direct on_stop or on_close server-side callback for cleanup.
# The connection will eventually time out or be closed when the Python process ends.
# For long-running applications or managed deployments, consider connection pooling or
# other strategies if connection management becomes an issue.
