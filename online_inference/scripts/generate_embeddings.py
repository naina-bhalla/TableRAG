"""
Generate embeddings from JSON docs under json_files/json_files and save a .pkl
that the retriever can load: a dict with keys 'embeddings', 'chunks', 'chunk_file_index'.

Usage:
  python online_inference/scripts/generate_embeddings.py

This script expects to run from the repository root.
"""
import os
import json
import pickle
import numpy as np
from tqdm import tqdm

# simple chunker by characters with overlap
def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    json_dir = os.path.join(repo_root, 'json_files', 'json_files')
    out_dir = os.path.join(repo_root, 'offline_data_ingestion_and_query_interface', 'embeddings')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'embedding.pkl')

    # late import to avoid requiring heavy deps at module import time
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print('Please install sentence-transformers in your venv before running this script')
        raise

    model = SentenceTransformer('all-MiniLM-L6-v2')

    chunks = []
    chunk_file_index = {}

    file_list = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
    file_list = sorted(file_list)

    for fname in tqdm(file_list, desc='Reading JSON files'):
        fpath = os.path.join(json_dir, fname)
        try:
            data = json.load(open(fpath, 'r', encoding='utf-8'))
        except Exception as e:
            print(f'Failed to read {fpath}: {e}')
            continue

        # each JSON file contains an array of tables; join titles/descriptions/data
        doc_text = ''
        for entry in data:
            title = entry.get('title', '')
            desc = entry.get('description', '')
            data_lines = entry.get('data', [])
            data_text = '\n'.join(data_lines) if isinstance(data_lines, list) else str(data_lines)
            doc_text += f"Title: {title}\nDescription: {desc}\n{data_text}\n\n"

        # chunk document
        doc_chunks = chunk_text(doc_text, chunk_size=1000, overlap=200)
        for c in doc_chunks:
            idx = len(chunks)
            chunks.append(c)
            chunk_file_index[idx] = fname

    print(f'Total chunks: {len(chunks)}')
    if len(chunks) == 0:
        print('No chunks found — aborting')
        return

    # encode in batches
    batch_size = 256
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc='Encoding'):
        batch = chunks[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False)
        embeddings.append(embs)

    embeddings = np.vstack(embeddings)

    data_out = {
        'embeddings': embeddings,
        'chunks': chunks,
        'chunk_file_index': chunk_file_index
    }

    with open(out_path, 'wb') as f:
        pickle.dump(data_out, f)

    print('Saved embeddings to', out_path)


if __name__ == '__main__':
    main()
