# YouTube Video Chatbot 🎥🤖

A Python-based RAG (Retrieval Augmented Generation) application that allows you to "chat" with any YouTube video. It fetches the video transcript, indexes it, and uses Google's Gemini LLM to answer your questions based on the video's content.

## 🚀 Features

* **Transcript Extraction:** Automatically fetches transcripts from YouTube videos using the video ID.
* **Smart Chunking:** Splits long transcripts into manageable chunks for processing.
* **Vector Embeddings:** Uses Hugging Face's `all-MiniLM-L6-v2` model to create efficient text embeddings locally.
* **Vector Search:** Utilizes FAISS (Facebook AI Similarity Search) for fast and accurate information retrieval.
* **AI-Powered Answers:** Integrates Google's **Gemini Pro** (via LangChain) to generate natural language responses based on the retrieved context.

## 🛠️ Tech Stack

* **Python**
* **[LangChain](https://www.langchain.com/):** Framework for building LLM applications.
* **[Google Gemini API](https://ai.google.dev/):** The Large Language Model (LLM) used for reasoning.
* **[Hugging Face](https://huggingface.co/):** For local embeddings (`sentence-transformers`).
* **[FAISS](https://github.com/facebookresearch/faiss):** For vector storage and similarity search.
* **[YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/):** To retrieve video subtitles.

## 📦 Installation

1. **Clone the repository** (if running locally) or open the notebook in Google Colab.
2. **Install the required dependencies:**
```bash
pip install youtube-transcript-api langchain-community langchain-google-genai \
            faiss-cpu python-dotenv langchain langchain-huggingface

```



## 🔑 Configuration

To run this project, you need a Google Gemini API Key.

1. Get your key from [Google AI Studio](https://aistudio.google.com/).
2. If using **Google Colab**:
* Add your key to the "Secrets" tab (key icon on the left) with the name `GEMINI_API_KEY`.


3. If running **locally**:
* Create a `.env` file and add:
```env
GEMINI_API_KEY=your_api_key_here

```





## 🏃‍♂️ How to Use

1. **Open the Notebook:** Run the `YouTube_Chatbot(improved).ipynb` file.
2. **Input Video ID:**
* Find the ID of the YouTube video you want to chat with.
* *Example:* For `https://www.youtube.com/watch?v=Gfr50f6ZBvo`, the ID is `Gfr50f6ZBvo`.
* Update the `video_id` variable in the code:
```python
video_id = "Gfr50f6ZBvo"

```




3. **Run the Cells:** Execute the cells to fetch the transcript, create the vector store, and initialize the chain.
4. **Ask Questions:**
* Use the final chain to ask questions about the video:
```python
response = final_chain.invoke("What is this video about?")
print(response)

```





## 🧠 How It Works (RAG Pipeline)

1. **Ingestion:** The script downloads the transcript text from YouTube.
2. **Splitting:** The text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding:** Each chunk is converted into a numerical vector using the `HuggingFaceEmbeddings` model.
4. **Storage:** Vectors are stored in a local `FAISS` index.
5. **Retrieval:** When you ask a question, the system searches FAISS for the most relevant transcript chunks.
6. **Generation:** The relevant chunks + your question are sent to the **Gemini LLM**, which generates a precise answer.

## 📝 Notes

* The project uses **Hugging Face embeddings** running locally, so you do not need an OpenAI API key for embeddings.
* Ensure the YouTube video has closed captions (CC) enabled (auto-generated or manual) for the transcript API to work.
