import argparse
import os
from pathlib import Path

from .embeddings import HashingEmbedder, TransformerEmbedder
from .rag import build_index, compose_grounded_answer, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A tiny Transformer/RAG learning project."
    )
    parser.add_argument("--docs", default="sample_docs", help="Directory of .md/.txt docs")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument(
        "--backend",
        choices=["transformer", "hashing"],
        default="transformer",
        help="Embedding backend",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model name for transformer backend",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached Hugging Face model files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Model cache directory. Use a non-C drive path such as D:\\hf-cache\\transformers.",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print the learning steps while the pipeline runs",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back to hashing when transformer loading fails",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    if args.show_steps:
        print("Step 1/4: Choose an embedding backend.")
    embedder = _make_embedder(args)

    if args.show_steps:
        print("Step 2/4: Read documents, split them into chunks, and embed each chunk.")
    index = build_index(
        docs_dir,
        embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    if args.show_steps:
        print(f"Indexed {len(index.chunks)} chunks from {docs_dir}.")
        print("Step 3/4: Embed the question and search for the nearest chunks.")
    results = retrieve(args.query, index, embedder, top_k=args.top_k)
    if args.show_steps:
        print("Step 4/4: Print a source-grounded answer.")
    print(compose_grounded_answer(args.query, results))


def _make_embedder(args):
    if args.backend == "hashing":
        print("Using hashing backend. This is for pipeline learning, not real semantics.")
        return HashingEmbedder()

    if not args.offline and args.cache_dir is None:
        message = (
            "No --cache-dir was provided, so the program will not download a "
            "Transformer model to the default cache. Use --cache-dir D:\\hf-cache\\transformers "
            "or add --offline if the model is already cached."
        )
        if args.no_fallback:
            raise SystemExit(message)
        print(message)
        print("Falling back to hashing backend so nothing is downloaded to C drive.")
        return HashingEmbedder()

    if args.cache_dir is not None:
        cache_error = _prepare_cache_environment(args.cache_dir)
        if cache_error:
            if args.no_fallback:
                raise SystemExit(cache_error)
            print(cache_error)
            print("Falling back to hashing backend so nothing is downloaded to C drive.")
            return HashingEmbedder()

    try:
        print(f"Loading Transformer model: {args.model}")
        return TransformerEmbedder(
            model_name=args.model,
            local_files_only=args.offline,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        if args.no_fallback:
            raise
        print("Transformer backend could not be loaded.")
        print(f"Reason: {exc}")
        print("Falling back to hashing backend so the pipeline can still run.")
        return HashingEmbedder()


def _prepare_cache_environment(cache_dir: Path) -> str:
    resolved = cache_dir.expanduser().resolve()
    if resolved.drive.upper() == "C:":
        return f"Refusing to use a C drive cache directory: {resolved}"

    cache_root = resolved.parent if resolved.name.lower() in {"models", "transformers"} else resolved
    os.environ.setdefault("HF_HOME", str(cache_root / "hf-home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    return ""


if __name__ == "__main__":
    main()
