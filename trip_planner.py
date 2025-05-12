import gradio as gr
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class PlannerState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    city: str
    interests: List[str]
    days: int
    itinerary: str

# Initialize LLM
lln = ChatGroq(
    temperature=0,
    groq_api_key="Enter Your API Key",  # <-- Replace with your working key
    model_name="llama-3.3-70b-versatile"
)

# Updated prompt with multi-day planning + transport suggestions
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful travel assistant. Create a detailed {days}-day travel itinerary for the city: {city}, based on these interests: {interests}. "
     "For each day, include time slots, location names, and suggested activities. Don't show Chokhi Dhani with City Palace, Jaigarh, Nahargarh and Amber Fort. Suggest famous restaurants, cafes and food stalls for meals and breaks during the travel."
     "Just after each day's itinerary, briefly suggest suitable modes of transport between locations (e.g., walk, metro, taxi), including links if possible."),
    ("human", "Please plan my trip.")
])

def travel_planner(city: str, interests: str, days: int) -> str:
    state = {
        "messages": [],
        "city": city,
        "interests": [i.strip() for i in interests.split(",") if i.strip()],
        "days": days,
        "itinerary": ""
    }

    # Add initial message for tracking
    state["messages"].append(
        HumanMessage(content=f"City: {state['city']}, Interests: {', '.join(state['interests'])}, Days: {state['days']}")
    )

    # Format prompt and call LLM
    try:
        formatted_prompt = itinerary_prompt.format_messages(
            city=state["city"],
            interests=", ".join(state["interests"]),
            days=state["days"]
        )
        response = lln.invoke(formatted_prompt)
        itinerary = response.content

        # Update state
        state["messages"].append(AIMessage(content=itinerary))
        state["itinerary"] = itinerary

        return itinerary

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=travel_planner,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
        gr.Number(label="Number of days", value=2)
    ],
    outputs=gr.Textbox(label="Multi-day itinerary with transport suggestions"),
    title="Multi-day Travel Planner",
    description="Plan a personalized trip for multiple days with activities, timings, and transport suggestions."
)

interface.launch()

