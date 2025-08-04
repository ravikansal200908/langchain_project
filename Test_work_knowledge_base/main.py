from file_manager import upload_files, list_uploaded_files, delete_file
from embedding_manager import ingest_documents, delete_chunks_by_filename
# from rag_chain import get_rag_chain
from rag_chain import run_rag_with_fallback


def main():
    while True:
        print("\n=== Knowledge Base Menu ===")
        print("1. Upload Files")
        print("2. View Uploaded Files")
        print("3. Delete File")
        print("4. Ingest Data to VectorDB")
        print("5. Delete Chunks by File Name")
        print("6. Ask a Question (RAG)")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            files = input("Enter file paths (comma-separated): ").split(",")
            upload_files([f.strip() for f in files])
        elif choice == "2":
            print(list_uploaded_files())
        elif choice == "3":
            file = input("Enter file name to delete: ")
            success = delete_file(file)
            delete_chunks_by_filename(file)
            print("Deleted" if success else "File not found")
        elif choice == "4":
            ingest_documents()
            print("Data ingested successfully")
        elif choice == "5":
            file = input("Enter file name to delete chunks for: ")
            delete_chunks_by_filename(file)
            print("Chunks deleted.")
        elif choice == "6":
            # chain = get_rag_chain()
            # query = input("Enter your question: ")
            # print("\nAnswer:", chain.run(query))

            query = input("Enter your question: ")
            answer = run_rag_with_fallback(query)
            print("\nAnswer:", answer)

        elif choice == "0":
            break


if __name__ == "__main__":
    main()
