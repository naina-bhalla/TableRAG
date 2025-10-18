from main import TableRAG
import argparse
import os

# Resolve repo root and create absolute defaults to avoid CWD issues
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DOC_DIR = os.path.join(REPO_ROOT, 'offline_data_ingestion_and_query_interface', 'embeddings')
DEFAULT_EXCEL_DIR = os.path.join(REPO_ROOT, 'offline_data_ingestion_and_query_interface', 'dataset', 'All India Tables')
DEFAULT_BGE_DIR = os.path.join(REPO_ROOT, 'bge_models')
DEFAULT_SAVE = os.path.join(REPO_ROOT, 'results', 'all_tables_output.jsonl')

_args = argparse.Namespace(
    backbone="cohere-command",
    data_file_path="online_inference/questions.jsonl",
    doc_dir=DEFAULT_DOC_DIR,
    excel_dir=DEFAULT_EXCEL_DIR,
    bge_dir=DEFAULT_BGE_DIR,
    save_file_path=DEFAULT_SAVE,
    max_iter=5,
    rerun=False
)

agent = TableRAG(_args)
agent.run(
    file_path=_args.data_file_path,
    save_file_path=_args.save_file_path,
    backbone=_args.backbone,
    rerun=_args.rerun
)
