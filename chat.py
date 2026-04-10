import os

import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from rag import build_rag_pipeline, extract_text_with_markitdown, process_pdf_with_ocr
from vlm import MinimoVLM


class MinimoInteractiveChat:
    """
    Small interactive wrapper around the text model, vision bridge, and RAG.

    The class keeps initialization costs in one place so the model is loaded
    once and then reused across many chat turns.
    """

    def __init__(self, model_path="./hf_minimo_merged"):
        print("Initializing Minimo interactive chat...")

        self.tokenizer_obj = Tokenizer.from_file("minimo_tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer_obj)
        self.tokenizer.pad_token = "<pad>"

        print("Loading model weights. This may take a few seconds...")
        self.vlm = MinimoVLM(hf_model_path=model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Float16 is used on GPU to save memory and speed up matrix operations.
        # CPU execution keeps the original dtype because half precision is not a
        # practical default there.
        if self.device == "cuda":
            self.vlm = self.vlm.to(torch.float16)
        self.vlm = self.vlm.to(self.device)
        self.vlm.eval()

    def _extract_file_paths(self, text):
        """
        Separate likely file paths from the natural-language part of the prompt.

        This lightweight parser allows drag-and-drop terminal usage without
        introducing a more complicated command grammar.
        """
        words = text.split()
        image_paths = []
        document_paths = []
        cleaned_words = []

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        document_extensions = [".pdf", ".md", ".txt"]

        for word in words:
            clean_word = word.strip("',\"")
            is_path = False

            if os.path.exists(clean_word):
                _, extension = os.path.splitext(clean_word.lower())
                if extension in image_extensions:
                    image_paths.append(clean_word)
                    is_path = True
                elif extension in document_extensions or os.path.isdir(clean_word):
                    document_paths.append(clean_word)
                    is_path = True

            if not is_path:
                cleaned_words.append(word)

        return " ".join(cleaned_words), image_paths, document_paths

    def _retrieve_rag_context(self, doc_paths, query):
        """
        Read local documents, build a temporary vector index, and retrieve text.

        The retrieval index is created on demand for the documents mentioned in
        the current turn. That keeps the interaction simple and avoids requiring
        a permanent ingestion workflow before chatting can start.
        """
        print(f"\n[RAG] Processing documents: {doc_paths}")
        all_text = ""

        for path in doc_paths:
            if path.endswith(".pdf"):
                ocr_path = path.replace(".pdf", "_ocr.pdf")
                processed_pdf = process_pdf_with_ocr(path, ocr_path)
                if processed_pdf:
                    all_text += extract_text_with_markitdown(processed_pdf) + "\n"
            elif path.endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8") as file_handle:
                    all_text += file_handle.read() + "\n"
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for filename in files:
                        if filename.endswith((".txt", ".md")):
                            file_path = os.path.join(root, filename)
                            with open(file_path, "r", encoding="utf-8") as file_handle:
                                all_text += file_handle.read() + "\n"

        if not all_text.strip():
            return ""

        print("[RAG] Searching for the most relevant passages...")

        # Retrieval uses embeddings only here. The text model that is already
        # loaded for chat remains the single generator to avoid loading a second
        # copy of the language model into memory.
        index = build_rag_pipeline(all_text, collection_name="chat_temp", use_minimo_llm=False)
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        return "\n".join(node.node.text for node in nodes)

    def chat(self, user_input):
        """
        Route one chat turn through text-only or image-aware generation.
        """
        query, image_paths, document_paths = self._extract_file_paths(user_input)

        context = ""
        if document_paths:
            context = self._retrieve_rag_context(document_paths, query)

        image = None
        if image_paths:
            print(f"\n[Vision] Loading image: {image_paths[0]}")
            image = Image.open(image_paths[0]).convert("RGB")
            if len(image_paths) > 1:
                print("[Vision] Only the first image is used in each turn.")

        print("\nMinimo is thinking...")

        if image is not None:
            with torch.no_grad():
                response = self.vlm.generate_with_image_and_rag(
                    image=image,
                    text_query=query,
                    retrieved_rag_context=context,
                    tokenizer=self.tokenizer_obj,
                )
            return response

        if context:
            full_prompt = f"Context: {context}\nUser: {query}\nMinimo:"
        else:
            full_prompt = f"User: {query}\nMinimo:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            output_ids = self.vlm.llm.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        response = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        return response.strip()


def main():
    print("==================================================")
    print(" Minimo Interactive Chat")
    print(" - Type a message to chat normally.")
    print(" - Include a file path to analyze it.")
    print(" - Supported files: .jpg, .png, .pdf, .md, .txt, or folders")
    print(" - Example: Summarize this file /home/david/notes.txt")
    print(" - Type 'exit' or 'quit' to close.")
    print("==================================================\n")

    chat_app = MinimoInteractiveChat()

    while True:
        try:
            user_input = input("\nMe: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            response = chat_app.chat(user_input)
            print(f"\nMinimo: {response}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
