# Web Scraping and AI Chatbot with LangChain and Google Vertex AI

This project combines web scraping, text processing, and AI-powered chatbot functionality using Python, LangChain, Gradio, and Google Cloud services.

## Prerequisites

1. **Google Cloud Setup**:
   - Enable the Vertex AI API and BigQuery.
   - Set up a dataset and table in BigQuery.
   - Install and configure the Google Cloud SDK.

2. **Python Environment**:
   - Python 3.8+
   - Install the required packages:
     ```bash
     pip install gradio langchain-google-community langchain-google-vertexai playwright
     ```
   - Set up Playwright:
     ```bash
     playwright install
     ```

3. **Environment Variables**:
   - Set your Google Cloud `PROJECT_ID` and ensure `GOOGLE_APPLICATION_CREDENTIALS` points to your service account key file.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-repo>/web-scraping-chatbot.git
   cd web-scraping-chatbot
   ```

2. **Set Up Configuration Variables**:
   Update the following variables in the script:
   ```python
   PROJECT_ID = "<your_project_id>"
   DATASET = "Scrape"
   TABLE = "scrapewebpage"
   REGION = "us-central1"
   MODEL_NAME = "textembedding-gecko@latest"
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

## Features

1. **Web Scraping**:
   - Extracts text content from a webpage using Playwright.
   - Saves data for processing and analysis.

2. **Text Processing**:
   - Splits scraped text into manageable chunks using LangChain.
   - Embeds text data using Vertex AI embeddings and stores it in BigQuery.

3. **AI-Powered Chatbot**:
   - Retrieves relevant text from BigQuery Vector Store based on user queries.
   - Generates responses using Google Vertex AI's Gemini model.

4. **User Interface**:
   - Gradio-based interface with tabs for scraping and chatting.

## How to Use

### Scraping and Storing Data
1. Navigate to the **Scrape and Process** tab.
2. Enter the URL you want to scrape.
3. Click "Scrape and Process" to store the processed data.

### Chatting with the AI
1. Navigate to the **Chat** tab.
2. Enter a query to interact with the chatbot.
3. Receive AI-generated responses based on the scraped data.

## Architecture

```text
+---------------------------------+
|         Gradio Interface        |
+---------------------------------+
     |             |
     v             v
+---------+   +-----------------+
| Scraper |   | Chat Interface  |
+---------+   +-----------------+
     |                 |
     v                 v
+-------------------------------+
|  LangChain Text Processing    |
+-------------------------------+
     |
     v
+-------------------------------+
|  BigQuery Vector Store        |
+-------------------------------+
     |
     v
+-------------------------------+
|  Google Vertex AI Embeddings  |
+-------------------------------+
```

## Notes

- Ensure your Google Cloud project is set up correctly.
- Use a robust User-Agent header when scraping.
- Follow website terms and conditions when scraping data.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contribution
Feel free to submit issues or pull requests to improve the project.
