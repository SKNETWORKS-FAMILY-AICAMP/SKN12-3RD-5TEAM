import os
import json
import re
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import INPUT_DIR, OUTPUT_DIR


def clean_text(raw_text: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw_text)
    text = re.sub(r"(그림\s?\d+[\.:]?\s?.*?$|표\s?\d+[\.:]?\s?.*?$|출처:.*?$)", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_all_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def load_and_process_json(filepath):
    try:
        with open(filepath, encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    raw_text = data.get("content", "")
    cleaned = clean_text(raw_text)

    filename = os.path.basename(filepath)
    if filename.startswith("13"):
        category = "교과서"
    elif filename.startswith("5"):
        category = "논문"
    else:
        category = "기타"

    return Document(
        page_content=cleaned,
        metadata={
            "filename": filename,
            "c_id": data.get("c_id", ""),
            "source": data.get("source_spec", ""),
            "year": data.get("creation_year", ""),
            "path": filepath,
            "type": category
        }
    )


def save_processed_text(doc: Document, base_dir=None):
    if base_dir is None:
        base_dir = OUTPUT_DIR

    category = doc.metadata.get("type", "기타")
    save_dir = os.path.join(base_dir, category)
    os.makedirs(save_dir, exist_ok=True)

    fname = f"{doc.metadata.get('c_id', 'unknown')}.txt"
    path = os.path.join(save_dir, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write(doc.page_content)


def load_all_documents(input_dir=INPUT_DIR):
    json_paths = find_all_json_files(input_dir)
    print(f"Found {len(json_paths)} JSON files.")

    docs = []
    for path in tqdm(json_paths, desc="Loading documents"):
        doc = load_and_process_json(path)
        if doc:
            save_processed_text(doc)
            docs.append(doc)

    return docs


def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "original_char_count": len(doc.page_content)
                }
            ))

    print(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks.")
    return all_chunks
