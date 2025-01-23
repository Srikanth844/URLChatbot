import os
import gradio as gr
from langchain_google_community import BigQueryVectorStore
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from playwright.sync_api import sync_playwright
from gradio import ChatMessage
# Set environment variable for gRPC
os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'

# Configuration variables
PROJECT_ID = ""
DATASET = "Scrape"
TABLE = "scrapewebpage"
REGION = "us-central1"
MODEL_NAME = "textembedding-gecko@latest"

# Initialize Google Cloud AI Platform
aiplatform.init(project=PROJECT_ID, location=REGION)
llm = VertexAI(model_name="gemini-pro")

# Initialize Vertex AI Embeddings object
embedding = VertexAIEmbeddings(
    model_name=MODEL_NAME,
    project=PROJECT_ID,
    location=REGION
)

# Initialize BigQuery Vector Store
store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
)

def scrape_webpage(url):
    save_dir = "scraped_data1"
    os.makedirs(save_dir, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        page.goto(url)
        
        # Wait for the body to load
        page.wait_for_selector('body')
        
        # Extract the text content
        content = page.inner_text('body')
        browser.close()
    return content

def process_and_store_data(content):
    # Split content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(content)
    
    # Create metadata for each text chunk
    metadatas = [{"len": len(t)} for t in splits]

    # Add text chunks and metadata to the vector store
    store.add_texts(splits, metadatas=metadatas)
    return "Data processed and stored successfully"

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])

def chatbot_response(message, history):
    docs = store.similarity_search(message)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Based on the following information, answer this query: {message}\n\nInformation: {context}"
    response = llm.predict(prompt)
    # history.append({"role": "user", "content": message})
    # history.append({"role": "assistant", "content": response})
    return {"role": "assistant", "content": response}

def scrape_and_process(url):
    content = scrape_webpage(url)
    result = process_and_store_data(content)
    return result

with gr.Blocks() as demo:
    gr.Markdown("# Web Scraping and AI Chatbot")
    
    with gr.Tab("Scrape and Process"):
        url_input = gr.Textbox(label="Enter URL to scrape")
        scrape_button = gr.Button("Scrape and Process")
        scrape_output = gr.Textbox(label="Processing Result")
        scrape_button.click(fn=scrape_and_process, inputs=url_input, outputs=scrape_output)
    
    with gr.Tab("Chat"):
        # chatbot = gr.Chatbot(type="messages")
        # chatbot.like(vote, None, None)
        gr.ChatInterface(
            fn=chatbot_response,
            title="AI Chatbot",
            description="Ask me anything about the data in the database!",
            cache_examples=True,
            save_history=True,
            flagging_mode="manual",
            type="messages"
        )

demo.launch()
