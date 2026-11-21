from app.services.agent import ClauseAgent


def main() -> None:
    clause_agent = ClassAgent()
    try:
        # Ask once at startup. Same thread_id = same conversation memory.
        user_id = input("User id: ").strip() or "guest"
        session = input("Conversation name: ").strip() or "default"
        thread_id = f"{user_id}:{session}"

        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "bye", "exit"):
                break

            result = clause_agent.ask(question=query, thread_id=thread_id, user_id=user_id)
            print(f"AI: {result['answer']}")

            if result["sources"]:
                pages = sorted({s["page"] for s in result["sources"] if s.get("page") is not None})
                print(f"    (pages: {pages})")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        clause_agent.close()


if __name__ == "__main__":
    main()