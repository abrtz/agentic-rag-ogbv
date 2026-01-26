from dotenv import load_dotenv

load_dotenv()


from graph.graph import app


def main():
    print("Hello Advanced RAG")
    print(
        app.invoke(
            input={
                # "question": "what forms of online violence are women facing online in 2025?"
                "question": "what is online violence against women?"
                # "question": "how to make pizza?"
            }
        )
    )


if __name__ == "__main__":
    main()
