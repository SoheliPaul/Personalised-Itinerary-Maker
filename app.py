import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
from langchain_groq import ChatGroq
from IPython.display import Markdown
from datetime import datetime
from io import StringIO

# Streamlit title
st.title("🌍 Travel Itinerary Generator ✈")
st.write("Welcome to the Travel Itinerary Generator! 🗺")
st.write("Provide us with the details, and we'll create a customized travel itinerary for you. 🌟")


# User Inputs
customer = st.text_input("👤 Customer Name:")
inquiry = st.text_area("📝 What do you want us to do? (e.g., Create a travel itinerary for my trip)")

#"I need help with creating a travel itinerary for my recent travel plans. Can you please provide guidance and curate an itinerary for me?

website_url = st.text_input("🔗 Enter the Travel Website URL:")

# Button to trigger the workflow
if st.button("Generate Itinerary 🚀"):

    # Initialize the ScrapeWebsiteTool with the user-provided URL
    tool = ScrapeWebsiteTool(website_url=website_url)
    text = tool.run()

    # Define the LLM Model (Llama3–70B)
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key= 'Groq_API_Key'
    )

    # Define the agents
    scraper = Agent(
        llm=llm,
        role="Website Scraper",
        goal="Scrape detailed and up-to-date information about tourism destinations from the specified website.",
        backstory="You are tasked with gathering content from a specific tourism website. "
                  "Your job is to collect data that will later be summarized and edited for a travel guide. "
                  "You focus on retrieving factual and detailed information about popular destinations, "
                  "including attractions, accommodations, and travel tips.",
        allow_delegation=False,
        verbose=True,
        tools=[tool]
    )

    summarizer = Agent(
        llm=llm,
        role="Content Summarizer",
        goal="Summarize the scraped tourism content into concise and useful travel information.",
        backstory="You receive raw data scraped from tourism websites. "
                  "Your job is to create a summarized version of this data, "
                  "condensing the most important information into a more digestible format. "
                  "This summarized content will help travelers quickly understand the essentials of each destination.",
        allow_delegation=False,
        verbose=True
    )

    itinerary_creator = Agent(
        llm=llm,
        role="Travel Itinerary Creator",
        goal="Create a personalized travel itinerary based on the summarized tourism information.",
        backstory="You are responsible for crafting a detailed travel itinerary for a customer "
                  "based on the summarized information about their chosen destinations. "
                  "Your task is to organize the trip, including accommodations, activities, and travel tips, "
                  "ensuring it aligns with the customer's preferences and interests.",
        allow_delegation=False,
        verbose=True
    )

    # Define the tasks
    scrape = Task(
        description=(
            "1. Identify the specific tourism website URL and scrape detailed information "
            "about attractions, accommodations, and travel tips.\n"
            "2. Ensure that the data collected is relevant, up-to-date, and comprehensive."
        ),
        expected_output="A collection of raw tourism information from the specified website, ready for summarization.",
        agent=scraper,
    )

    summarize = Task(
        description=(
            "1. Summarize the scraped content to highlight the key attractions, accommodations, and tips for each destination.\n"
            "2. Ensure the summaries are concise and informative.\n"
            "3. Organize the content logically for easy reading."
        ),
        expected_output="A set of concise summaries for each travel destination, ready for itinerary creation.",
        agent=summarizer,
        context = [scrape]
    )

    create_itinerary = Task(
        description=(
            "1. Use the summarized tourism information to create a personalized travel itinerary.\n"
            "2. Include recommended accommodations, daily activities, and key travel tips.\n"
            "3. Ensure the itinerary is well-organized, matching the customer's preferences and interests."
        ),
        expected_output="A detailed travel itinerary tailored to the customer's preferences.",
        agent=itinerary_creator,
        context = [summarize]
    )

    # Create a Crew
    crew = Crew(
        agents=[scraper, summarizer, itinerary_creator],
        tasks=[scrape, summarize, create_itinerary],
        verbose=True)

    # Define inputs
    inputs = {
        "customer": customer,
        "inquiry": inquiry
    }

    # Kick off the workflow
    result = crew.kickoff(inputs=inputs)
    st.write(f"🎉 Travel Itinerary For {customer} 🎉")
    # Display the result in a more styled forma

    # Display the result in Markdown format
    st.markdown(result, unsafe_allow_html=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"travel_itinerary_{timestamp}.txt"

    # Create a downloadable file for the result
    result_str = str(result) 
    st.download_button(
        label="Download Itinerary 📥",
        data=result_str,
        file_name=file_name,
        mime='text/plain'
    )