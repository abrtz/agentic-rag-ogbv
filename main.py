from typing import Any, Dict, List

import streamlit as st

from backend import run_llm

st.set_page_config(
    page_title="RespectHer: Information on Online Gender-based Violence against Women and Girls",
    layout="centered",
)
st.title(
    ":violet[RespectHer]: Information on Online Gender-based Violence against Women and Girls"
)

# ---sidebar---
with st.sidebar:
    # add google icons
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
    <style>
    .material-symbols-outlined {
        font-variation-settings:
            'FILL' 0,
            'wght' 400,
            'GRAD' 0,
            'opsz' 20;
        vertical-align: middle;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # subheader with icon
    st.markdown(
        """
    <h3 style="color:#5B197B;">
        <span class="material-symbols-outlined">smart_toy</span>
        Chat Session
    </h3>
    """,
        unsafe_allow_html=True,
    )
    # st.subheader(":violet[Chat session]")
    if st.button("Clear chat", use_container_width=True):
        # if button is clicked, clear chat (messages in session state)
        st.session_state.pop("messages", None)
        # refresh app
        st.rerun()

    st.markdown("---")

    # add info about the app
    st.markdown(
        """
    <h3 style="color:#5B197B;">
        <span class="material-symbols-outlined">info</span>
        About
    </h3>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""**RespectHer** is an agentic RAG application that provides information about
    online violence against women and girls, the manosphere, deepfakes, 
    laws and regulations related to gender-based violence,
    and measures to mitigate gender-based violence.
    """
    )
    st.markdown("Author: [Ariana Britez](https://github.com/abrtz)")

    st.markdown("---")
    st.markdown(
        """
    <h3 style="color:#5B197B;">
        <span class="material-symbols-outlined">library_books</span>
        Resources
    </h3>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    - [UN Women Articles](https://www.unwomen.org/en/articles/)
    - [UN Women News](https://www.unwomen.org/en/news-stories/)
    - [Report](https://www.unwomen.org/sites/default/files/2025-06/normative-advances-on-technology-facilitated-violence-against-women-and-girls-en.pdf): Normative Advances on Technology-Facilitated Violence Against Women and Girls
    - [Paper](https://www.isdglobal.org/wp-content/uploads/2023/09/Misogynistic-Pathways-to-Radicalisation-Recommended-Measures-for-Platforms-to-Assess-and-Mitigate-Online-Gender-Based-Violence.pdf): Misogynistic Pathways to Radicalisation
    - Web search by [Taviliy](https://www.tavily.com)
    """
    )

# ---initialize session state----

# write placeholder message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about online violence against women and girls. I'll retrieve relevant documents and cite sources.",
            "sources": [],
        }
    ]

# display sources
for msg in st.session_state.messages:
    # create a container to hold the message
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                # show the sources
                for s in msg["sources"]:
                    st.markdown(f"- {s}")  # dash is displayed as list item in markdown


# option quesitons
button_prompt = None

st.markdown("**Try one of these questions:**", unsafe_allow_html=True)
questions = [
    "How is AI amplifying violence against women and girls?",
    "What global and regional normatives have been implemented to fight online violence against women and girls?",
    "What is the manosphere and what terms are related to it?",
]

col1_button, col2_button, col3_button = st.columns([1, 1, 1])

# add clickable container for each question
for i, col in enumerate([col1_button, col2_button, col3_button]):
    with col:
        # make whole container clickable
        if st.button(
            questions[i],
            key=f"question_button_{i}",
            use_container_width=True,
            help="Click here to submit this question",
        ):
            button_prompt = questions[i]


# create text area where user can input message
user_input = st.chat_input("Or type your own question...")

# populate prompt with selected question or user input
prompt = button_prompt if button_prompt else user_input

# process prompt
if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "sources": [],
        }
    )  # append to session state
    with st.chat_message("user"):  # user message
        st.markdown(prompt)  # display it in markdown

    with st.chat_message("assistant"):  # assistant message, response of LLM
        # running RAG agent. if it fails, still display sth to the user
        try:
            with st.spinner(
                "Retrieving docs and generating answer..."
            ):  # show the user that sth is happening
                result: Dict[str, Any] = run_llm(prompt)
                answer = (
                    str(result.get("generation", "")).strip() or "(No answer returned.)"
                )
                sources = result.get("source", [])

            # show answer to user
            st.markdown(answer)
            # if there are sources, print them in markdown as list
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # save assistant response to session state
            st.session_state.messages.append(
                {
                    "role": "assistant",  # result of agent
                    "content": answer,  # answer of the agent
                    "sources": sources,  # sources variable as list
                }
            )
        # handle errors
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
