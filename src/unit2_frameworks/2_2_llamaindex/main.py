import asyncio

import chromadb
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings


def call_hf_model():
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.7,
        max_tokens=100,
        token=settings.HF_TOKEN,
    )

    res = llm.complete("Hello, how are you?")
    print(res)


async def sdr():
    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_overlap=0),
            HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5"),  # noqa: F821
        ]
    )

    nodes = await pipeline.arun(documents=[Document.example()])

    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )


async def main():
    # call_hf_model()
    await sdr()


if __name__ == "__main__":
    asyncio.run(main())
