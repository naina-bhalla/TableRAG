import os
import time
import urllib3
import sys
import faiss
import json
import transformers
import warnings
import threading
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tqdm as tqdm
from collections import defaultdict
from transformers import AutoModel
import transformers
from utils.tool_utils import *
import nltk
from more_itertools import chunked
import numpy as np
import pickle
from typing import Dict, List, Union, Tuple, Any
from utils.utils import read_plain_csv

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
device = "cuda:0"

class SemanticRetriever :
    """
    Retrieving process, containing recall and rerank.
    """
    def __init__(
        self,
        chunks: List[str],
        chunk_index: Dict,
        chunk_file_index: Dict = None,
        llm_path: str = None,
        reranker_path: str = None,
        save_path: str = "./retrieval_result/embedding.pkl"
    ) -> None:
        # If embeddings are precomputed, load them first and avoid creating heavy models
        # (this prevents unnecessary model downloads or local path validation errors)
        if os.path.exists(save_path):
            doc_embeddings, self.chunks, self.chunk_file_index = self.load_embeddings(save_path)
            self.chunk_index = {idx: ch for idx, ch in enumerate(self.chunks)}
            # Only create embedder/reranker if we need to embed more documents later
            self.embedding_model = None
            self.reranker = None
        else:
            # No precomputed embeddings: instantiate models and compute embeddings
            self.embedding_model = Embedder(llm_path) if llm_path else None
            self.reranker = Reranker(reranker_path) if reranker_path else None

            self.chunks = chunks
            self.chunk_index = chunk_index
            self.chunk_file_index = chunk_file_index
            doc_embeddings = self.embed_doc(chunks, save_path=save_path)
        
        self.thread_local = threading.local()
        self.index_lock = threading.RLock()

        print("embedding size", doc_embeddings.shape)
        # Try to use GPU resources if available; otherwise fall back to CPU index
        try:
            self.res = faiss.StandardGpuResources()
            self.use_gpu = True
        except Exception:
            # faiss may be the CPU-only build without GPU support
            self.res = None
            self.use_gpu = False

        self.index_IP = self.build_index(doc_embeddings)

    def embed_doc(self, chunks: List[str], batch_size: int = 512, save_path: str = None) -> Any :
        """
        Embed documents in batches for improved performance

        Args:
            chunks: List of text chunks to embed

        Returns:
            np.ndarray: Array of document embeddings
        """
        encode_vecs = []
        iterator = tqdm(range(0, len(chunks), batch_size)) if len(chunks) >= 100 \
            else range(0, len(chunks), batch_size)

        for i in iterator :
            batch = chunks[i: i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)

            if len(batch) == 1 :
                encode_vecs.append(batch_embeddings)
            else :
                encode_vecs.extend(batch_embeddings)
                 
        encode_vecs = np.array(encode_vecs)
        if len(encode_vecs.shape) == 3 :
            encode_vecs = encode_vecs.reshape(-1, encode_vecs.shape[-1])
        
        if save_path :
            self.save_embeddings(encode_vecs, chunks, save_path)
            print("Embedding Vectors Saved.")
        
        return encode_vecs
    
    def save_embeddings(self, embeddings: Any, chunks: List[str], save_path: str) -> None :
        """
        Save embeddings and optionally the original chunks.

        Args:
            embeddings: numpy array of embeddings
            chunks: orignal text chunks
            save_path: path to save the embeddings
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.pkl') :
            data = {
                "embeddings": embeddings,
                "chunks": chunks,
                "chunk_file_index": self.chunk_file_index
            }
            with open(save_path, "wb") as f :
                pickle.dump(data, f)
            print(f"Embeddings and chunks saved to {save_path}")

    @staticmethod
    def load_embeddings(load_path: str = None) -> Tuple[Any, Any] :
        """
        Load saved embeddings.

        Args:
            load_path: Path to the saved embeddings.
        
        Returns:
            embeddings if .npy file, (embeddings, chunks) if .pkl file
        """
        if load_path.endswith('.npy') :
            return np.load(load_path)
        elif load_path.endswith('.pkl') :
            with open(load_path, 'rb') as f :
                data = pickle.load(f)
            return data['embeddings'], data['chunks'], data['chunk_file_index']
        
    def build_index(self, dense_vector: Any) -> Any :
        print("Building Index.")
        with self.index_lock :
            _, dim = dense_vector.shape
            # CPU index (works for all installs)
            index_cpu = faiss.IndexFlatIP(dim)
            index_cpu.add(dense_vector)

            # If GPU is available and faiss provides GPU helpers, convert to GPU index
            if getattr(self, 'use_gpu', False) and hasattr(faiss, 'index_cpu_to_gpu'):
                try:
                    co = faiss.GpuClonerOptions()
                    # use device 0 by default
                    index_gpu = faiss.index_cpu_to_gpu(self.res, 0, index_cpu, co)
                    return index_gpu
                except Exception as e:
                    print(f"⚠️ Faiss GPU conversion failed, falling back to CPU index: {e}")
                    return index_cpu

            return index_cpu

    def retrieve(self, query, recall_num, rerank_num) :
        docs, ori_file_name = self.recall(query, recall_num)
        reranked_docs, rerank_scores, filenames = self.rerank(query, docs, rerank_num, ori_file_name)
        return reranked_docs, rerank_scores, filenames
        
    def recall(self, query: str, topn:int) -> List[str] :
        query_emb = self.embed_doc(query)
        with self.index_lock :
            D, I = self.index_IP.search(query_emb, topn)
        ori_docs = [self.chunk_index[i] for i in I[0]]
        ori_file_name = [self.chunk_file_index[i] for i in I[0]]
        return ori_docs, ori_file_name

    def rerank(self, query: str, docs: List[str], topn: int, ori_file_name: List[str]) -> Tuple[List[str], List[int]] :
        pairs = [[query, d] for d in docs]
        scores = self.reranker.compute_score(pairs)
        sroted_pairs = sorted(zip(scores, docs, ori_file_name), reverse=True)
        score_sorted, doc_sorted, filename_sorted = zip(*sroted_pairs)
        return doc_sorted[:topn], score_sorted[: topn], filename_sorted[: topn]


class MixedDocRetriever :
    def __init__(
        self,
        doc_dir_path: str,
        excel_dir_path: str,
        llm_path: str = None,
        reranker_path: str = None,
        save_path: str = "./retrieve_result/embedding.pkl"
    ) -> None:
        self.ori_documents = self.load_hybrid_dataset(doc_dir_path, excel_dir_path)
        print("Loading done.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunking_dict = self.doc_chunking()
        self.chunks, self.chunk_to_index, self.chunk_to_filename = self.build_index(doc_chunking_dict)
        self.semantic_retriever = SemanticRetriever(
            chunks=self.chunks,
            chunk_index=self.chunk_to_index,
            chunk_file_index=self.chunk_to_filename,
            llm_path=llm_path,
            reranker_path=reranker_path,
            save_path=save_path
        )

    def load_hybrid_dataset(self, doc_dir_path: str, excel_dir_path: str) -> Dict[str, List[str]] :
        all_docs = defaultdict(list)
        # Only consider supported Excel file extensions. Skip other files (e.g., .json) that may be present.
        supported_exts = ('.xlsx', '.xls', '.xlsm', '.xltx', '.xltm')
        for file in tqdm(os.listdir(excel_dir_path)) :
            if not file.lower().endswith(supported_exts):
                # skip non-excel files
                # print a short message for visibility
                # (useful when directories contain mixed file types)
                # tqdm will already show progress, so keep this minimal
                # but still helpful during troubleshooting
                # e.g., 'table_1_children_enrolled_in.json' leaked into the folder
                # which would cause openpyxl.InvalidFileException
                # We choose to skip and continue.
                #
                # Note: we intentionally don't raise here.
                #
                continue
            file_path = os.path.join(excel_dir_path, file)
            try:
                content = excel_to_markdown(file_path)
            except Exception as e:
                print(f"⚠️ Skipping file {file}: {e}")
                continue
            excel_content = content
            all_docs[file] = excel_content
        
        # Only load JSON/JSONL files from doc_dir; skip binaries (e.g., embedding.pkl)
        for file in tqdm(os.listdir(doc_dir_path)) :
            if not file.lower().endswith(('.json', '.jsonl')):
                continue
            file_path = os.path.join(doc_dir_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as fin :
                    data_split = json.load(fin)
            except UnicodeDecodeError:
                print(f"⚠️ Skipping non-utf8 file in doc_dir: {file}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
                continue

            key_value_doc = ''
            for key, item in data_split.items() :
                key_value_doc += f"{key} {item}\n"

            all_docs[file] = key_value_doc
        return all_docs


    def build_index(self, chunking_dict: Dict) -> Tuple :
        flatten_chunks = []
        chunk_to_index = defaultdict(list)
        chunk_to_filename = defaultdict(str)
        cnt = 0
        for idx, (key, item) in enumerate(chunking_dict.items()) :
            flatten_chunks += item
            for i in item :
                chunk_to_index[cnt] = i
                chunk_to_filename[cnt] = key
                cnt += 1
        return flatten_chunks, chunk_to_index, chunk_to_filename

    def nltk_single_doc_chunking(self, doc: str, key: str) -> Tuple[List[str], str] :
        all_splits = self.text_splitter.split_text(doc)
        add_file_name_splits = []
        for split_chunk in all_splits :
            add_file_name_chunk = f"File name: {key}\n" + split_chunk
            add_file_name_splits.append(add_file_name_chunk)
        return add_file_name_splits, key

    def doc_chunking(self, max_workers=10) -> Dict :
        doc_chunkings = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor :
            future_to_case = {executor.submit(self.nltk_single_doc_chunking, doc, key): doc for key, doc in self.ori_documents.items()}
            for future in tqdm(as_completed(future_to_case), total=len(self.ori_documents), desc="Processing cases") :
                try :
                    single_doc_chunking, key = future.result()
                    doc_chunkings[key] += single_doc_chunking
                except Exception as e :
                    print(f"Case processing generated exception: {e}")
        return doc_chunkings

    def retrieve(self, query: str, recall_nun: int = 50, rerank_num: int = 5) :
        return self.semantic_retriever.retrieve(query, recall_nun, rerank_num)





