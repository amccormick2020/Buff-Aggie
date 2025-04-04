import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from basic_flow import app, config


# Streamlit app title
st.title("Workout Plan Recommender")

# Sidebar for user preferences
st.sidebar.header("User Preferences")

# Main input area
st.header("Chat with the Workout Recommender")
user_input = st.text_input("Ask a question or request a workout plan:", "")

# Button to send the query
if st.button("Submit"):
    if user_input:
        # Prepare input for the chatbot
        input_message = [HumanMessage(content=user_input)]
        

        try:
            # Invoke the chatbot logic
            for input_m in input_message:
                output = app.invoke({"messages":input_m},config=config)
                output["messages"][-1].pretty_print()
           
            workout_plan = output["messages"][-1].content  # Extract the chatbot's response

            # Display the response
            st.subheader("Workout Plan")
            st.json(workout_plan)  # Display the JSON response in a readable format
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")
