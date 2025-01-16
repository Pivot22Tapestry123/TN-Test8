
import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chat_models import AzureChatOpenAI
import openai

# Streamlit UI for Azure credentials
st.sidebar.header("Azure Configuration")
azure_api_key = st.sidebar.text_input("Enter Azure API Key", type="password")

# Rest of the configuration
azure_api_base = "https://rstapestryopenai2.openai.azure.com/"
azure_api_version = "2023-05-15"

# Main app content
st.title("CrewAI with Azure OpenAI GPT-4")

# Only show the main app if API key is provided
if azure_api_key:
    # Configure OpenAI API settings for Azure
    openai.api_type = "azure"
    openai.api_version = azure_api_version
    openai.api_base = azure_api_base
    openai.api_key = azure_api_key

    # Create Azure OpenAI instance
    llm = AzureChatOpenAI(
        openai_api_key=azure_api_key,
        openai_api_base=azure_api_base,
        openai_api_version=azure_api_version,
        deployment_name="gpt-4",
        openai_api_type="azure",
        temperature=0.7
    )

    # Define the agent
    agent = Agent(
        role="Assistant",
        goal="Provide helpful responses to user queries.",
        backstory="This agent was designed to provide helpful, friendly, and accurate answers.",
        llm=llm,
        verbose=True
    )

    # Main query input
    user_input = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_input:
            # Define the task
            task = Task(
                description=user_input,
                agent=agent,
                expected_output="A helpful response to the user's query."
            )

            # Create and run the crew
            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()

            # Display the result
            st.subheader("Response:")
            st.write(result)
        else:
            st.warning("Please enter a query.")
else:
    st.warning("Please enter your Azure API Key in the sidebar to continue.")