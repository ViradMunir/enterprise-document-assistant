# Enterprise Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a collection of PDF documents and receive answers grounded in the retrieved document content.

The application uses **LangChain** for document processing and retrieval, **Hugging Face Sentence Transformers** for embeddings, **ChromaDB** as the vector database, **Groq** for LLM inference, and **Streamlit** for the user interface.

## Features

* Load and process PDF documents from a local `data/` directory
* Extract text from PDF pages using `PyPDFLoader`
* Split documents into smaller chunks for retrieval
* Generate vector embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model
* Store and retrieve document embeddings using ChromaDB
* Retrieve the top 3 most relevant document chunks for a question
* Generate answers using the Groq-hosted Llama 3.1 8B Instant model
* Restrict answers to information contained in the retrieved context
* Display the source PDF and page number used for the answer
* Automatically create the ChromaDB database when it does not already exist
* Rebuild the vector database from the documents using the **Rebuild Database** button

## How It Works

The application follows a standard Retrieval-Augmented Generation pipeline:

```text
PDF Documents
     │
     ▼
PDF Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
Hugging Face Embeddings
     │
     ▼
ChromaDB Vector Database
     │
     ▼
User Question
     │
     ▼
Similarity Retrieval
     │
     ▼
Top 3 Relevant Chunks
     │
     ▼
Prompt with Retrieved Context
     │
     ▼
Groq / Llama 3.1 8B Instant
     │
     ▼
Generated Answer + Sources
```

### 1. Document Loading

PDF files placed inside the `data/` directory are loaded using LangChain's `PyPDFLoader`.

Each PDF page is treated as a document, and its source filename and page number are stored as metadata.

### 2. Text Chunking

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

* **Chunk size:** 1000 characters
* **Chunk overlap:** 200 characters

The overlap helps preserve context between neighboring chunks.

### 3. Embeddings

Each document chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings allow the application to perform semantic similarity-based retrieval.

### 4. Vector Database

The embeddings are stored locally in **ChromaDB**.

The database is stored in:

```text
chroma_db/
```

If the directory does not exist, the application creates the database automatically from the documents in the `data/` directory.

The `chroma_db/` directory is intentionally excluded from version control because it is generated locally.

### 5. Retrieval

When a user enters a question, the application retrieves the **3 most relevant document chunks** from ChromaDB.

```python
retriever = vectordb.as_retriever(
    search_kwargs={
        "k": 3
    }
)
```

### 6. Answer Generation

The retrieved chunks are provided as context to the Llama 3.1 8B Instant model through the Groq API.

The model is instructed to:

1. Answer only from the provided context.
2. Avoid inventing information.
3. State that relevant information could not be found when the context is insufficient.

This helps reduce unsupported or hallucinated answers.

### 7. Source Attribution

The application also displays the PDF filename and page number associated with the retrieved documents.

This allows users to identify where the information used to generate the answer came from.

## Project Structure

```text
enterprise-document-assistant/
│
├── app.py                 # Streamlit application
├── data/                  # Add PDF documents here
│   └── .gitkeep
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment configuration
├── .gitignore             # Files excluded from Git
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

The following are generated locally and are **not committed to the repository**:

```text
chroma_db/
.env
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-document-assistant.git
cd enterprise-document-assistant
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The `.env` file should **never be committed to GitHub**.

### 5. Add PDF documents

Place your PDF files inside:

```text
data/
```

For example:

```text
data/
├── document1.pdf
├── document2.pdf
└── document3.pdf
```

The PDF documents used during development are not included in this repository. Add your own PDF documents to the `data/` directory.

### 6. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be accessed.

## Rebuilding the Database

If you add, remove, or replace PDF documents in the `data/` directory, click:

**Rebuild Database**

The application will:

1. Delete the existing ChromaDB database.
2. Clear the cached vector database.
3. Re-run the document loading and embedding process.
4. Create a new vector database from the current PDFs.

## Technologies Used

* **Python** — Application development
* **Streamlit** — Web interface
* **LangChain** — Document processing and retrieval pipeline
* **PyPDF** — PDF text extraction
* **Hugging Face Sentence Transformers** — Text embeddings
* **ChromaDB** — Vector database
* **Groq** — LLM inference
* **Llama 3.1 8B Instant** — Language model
* **python-dotenv** — Environment variable management

## Important Notes

### PDF Content

The current implementation uses `PyPDFLoader`, so the RAG pipeline primarily works with **text extracted from PDFs**. It does not perform multimodal understanding of images, charts, or diagrams contained in the PDFs.

### Documents

The PDF documents used during development are not included in this repository. To use the application, add your own PDF documents to the `data/` directory.

### API Key Security

Never upload your `.env` file or expose your Groq API key in source code.

Use environment variables instead:

```env
GROQ_API_KEY=your_api_key
```

### Vector Database

The ChromaDB database is generated locally from the documents and therefore does not need to be included in the repository.

## Future Improvements

Potential improvements include:

* Support for additional document formats
* Improved retrieval using similarity score thresholds
* Hybrid keyword + semantic search
* Reranking retrieved documents
* Conversation history
* Streaming LLM responses
* Better source citations
* Support for tables, images, and charts in PDFs
* Document upload directly through the Streamlit interface
* Deployment to a cloud platform

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
