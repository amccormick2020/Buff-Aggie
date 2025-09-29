# Aggie workout plan generator and exercise recommender
# Link to Web App Deployment Below
https://csce670.streamlit.app/

Buff Aggie utilizes LangChain and LangGraph to orchestrate the workflow among LLM agents and retrieval/recommendation algorithms. The user's input prompt is fed into an intent node, where a LLM determines the user's intent and reconstructs the prompt for similarity retrieval if the user requests targeted exercises. LangGraph will route the query to a ranked retrieval node where SBERT is used to retrieve relevant exercises if the user requests exercises with keywords, such as targeted areas of the body. Unless otherwise stated, recommended exercises will be based on a combination of saved user preferences and the user's input prompt. If the user simply requests a workput plan without specifics in the input prompt, recommendations are performed based on preferences the user saved in their profile. Simple user profiles with preferences are maintained using SQLite.
