import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from logic import get_thrust_chamber_params,MODEL

memory = MemorySaver()
agent_ = create_react_agent(MODEL.openai_client,
                            tools=[get_thrust_chamber_params],
                            checkpointer=memory)

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': "system", "content": """This is a Lazy Rocketeer agent (a part of 
    Gigala software) to reason around system requirements and mission parameters to design a rocket engine. When helping
     in design, it uses paradigm: think, act, observe and considers the following aspects:
     
    -   Decisions on basic parameters  
    -   Stage masses and thrust level  
    -   Propellant flows and dimensions of thrust chamber  
    -   Heat transfer  
    -   Injector design  
    -   Igniter dimensions  
    -   Layout drawings, masses, flows, and pressure drops"""}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello, how can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

with st.chat_message("assistant"):

    stream = agent_.invoke(
        {"input": prompt, "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]},
        {
            # "callbacks":[get_streamlit_cb(st.empty())],
            "configurable": {"thread_id": "abc321"},
        },
    )

    response = list(stream["messages"][len(stream["messages"])-1])[0][1]
    st.write(response)

st.session_state.messages.append({"role": "assistant", "content": response})
