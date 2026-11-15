from app.services.chat import RAGService


def main() -> None:
    rag_service = RAGService()
    try:
        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            try:
                result = rag_service.ask(question=query)
            except ValueError as e:
                print(f"Error: {e}")
                continue
            except Exception as e:
                print(f"Something went wrong: {e}")
                continue

            print(f"AI: {result.answer}")

            if result.sources:
                print("Sources:")
                for s in result.sources:
                    print(f"  - {s.source} (page {s.page}, score {s.score})")
            print()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        rag_service.close()


if __name__ == "__main__":
    main()

# cd api
# uv run python -m app.cli