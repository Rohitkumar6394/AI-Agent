import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Travel Agent",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 AI Travel Agent using LangGraph")
st.write("Ask your travel question and get an AI-generated travel plan.")


# =========================
# LOAD AI MODEL
# =========================

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


def generate_ai_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
    **inputs,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.8

 )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def is_bad_ai_answer(answer):
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()

    # Empty or very small answer
    if len(answer_clean) < 80:
        return True

    # If model repeats the prompt
    bad_phrases = [
        "you are an ai travel agent",
        "create a helpful travel plan",
        "create budget travel suggestions",
        "give trip summary",
        "explain why budget is low",
        "destination:",
        "hotel type:",
        "transport:",
        "estimated cost:"
    ]

    for phrase in bad_phrases:
        if phrase in answer_lower:
            return True

    # Repeated weak answer
    if answer_lower.count("budget is low") >= 2:
        return True

    if answer_lower.count("budget") >= 5 and len(answer_clean) < 180:
        return True

    # If no real plan words
    useful_words = ["day", "suggestion", "plan", "travel", "food", "advice"]
    if not any(word in answer_lower for word in useful_words):
        return True

    return False

# =========================
# LANGGRAPH STATE
# =========================

class TravelState(TypedDict):
    user_question: str
    destination: str
    days: int
    budget: int
    hotel_type: str
    transport: str
    estimated_cost: int
    status: str
    final_answer: str


# =========================
# HELPER FUNCTIONS
# =========================

def create_day_plan(destination, days):
    plan = []

    places = {
        "Goa": [
            "Baga Beach aur Calangute Beach visit karo.",
            "Fort Aguada aur local market explore karo.",
            "Dudhsagar Falls ya South Goa sightseeing karo.",
            "Beach activities aur sunset enjoy karo.",
            "Shopping aur return journey."
        ],
        "Manali": [
            "Mall Road aur local market visit karo.",
            "Hadimba Temple aur Vashisht Temple visit karo.",
            "Solang Valley sightseeing karo.",
            "Old Manali cafes aur nature walk enjoy karo.",
            "Shopping aur return journey."
        ],
        "Jaipur": [
            "Hawa Mahal aur City Palace visit karo.",
            "Amber Fort aur Jal Mahal explore karo.",
            "Local market aur Rajasthani food try karo.",
            "Nahargarh Fort sunset view enjoy karo.",
            "Shopping aur return journey."
        ],
        "Delhi": [
            "India Gate aur Connaught Place visit karo.",
            "Red Fort aur Chandni Chowk explore karo.",
            "Qutub Minar aur Lotus Temple visit karo.",
            "Local food aur shopping enjoy karo.",
            "Return journey."
        ],
        "Mumbai": [
            "Gateway of India aur Marine Drive visit karo.",
            "Juhu Beach aur Bandra explore karo.",
            "Siddhivinayak Temple aur local market visit karo.",
            "Film City ya shopping plan karo.",
            "Return journey."
        ],
        "Agra": [
            "Taj Mahal visit karo.",
            "Agra Fort aur Mehtab Bagh explore karo.",
            "Local market aur petha try karo.",
            "Nearby sightseeing karo.",
            "Return journey."
        ],
        "Unknown": [
            "Local sightseeing karo.",
            "Popular tourist places visit karo.",
            "Local food try karo.",
            "Shopping aur nearby attractions explore karo.",
            "Return journey."
        ]
    }

    destination_places = places.get(destination, places["Unknown"])

    for i in range(1, days + 1):
        if i == 1:
            activity = f"Arrival, hotel check-in, {destination_places[0]}"
        elif i == days:
            activity = f"Shopping, packing aur return journey."
        else:
            activity = destination_places[(i - 1) % len(destination_places)]

        plan.append(f"**Day {i}:** {activity}")

    return "\n\n".join(plan)


# =========================
# NODE 1: EXTRACT DETAILS
# =========================

def extract_details(state: TravelState):
    question = state["user_question"].lower()

    destinations = ["goa", "manali", "jaipur", "delhi", "mumbai", "agra"]
    destination = "Unknown"

    for city in destinations:
        if city in question:
            destination = city.title()
            break

    days = 3
    days_match = re.search(r"(\d+)\s*(day|days|din)", question)

    if days_match:
        days = int(days_match.group(1))

    numbers = re.findall(r"\d+", question)
    numbers = [int(num) for num in numbers]

    budget = 10000

    if numbers:
        large_numbers = [num for num in numbers if num >= 1000]
        if large_numbers:
            budget = max(large_numbers)

    if "luxury" in question:
        hotel_type = "Luxury"
    elif "standard" in question or "normal" in question:
        hotel_type = "Standard"
    else:
        hotel_type = "Budget"

    if "flight" in question:
        transport = "Flight"
    elif "cab" in question or "taxi" in question:
        transport = "Cab"
    elif "bus" in question:
        transport = "Bus"
    else:
        transport = "Train"

    state["destination"] = destination
    state["days"] = days
    state["budget"] = budget
    state["hotel_type"] = hotel_type
    state["transport"] = transport

    return state


# =========================
# NODE 2: CALCULATE COST
# =========================

def calculate_cost(state: TravelState):
    destination_cost = {
        "Goa": 3500,
        "Manali": 3000,
        "Jaipur": 2500,
        "Delhi": 2200,
        "Mumbai": 4000,
        "Agra": 2000,
        "Unknown": 2500
    }

    hotel_cost = {
        "Budget": 1000,
        "Standard": 2000,
        "Luxury": 4000
    }

    transport_cost = {
        "Bus": 1000,
        "Train": 1500,
        "Flight": 5000,
        "Cab": 3000
    }

    place_cost = destination_cost.get(state["destination"], 2500)
    hotel = hotel_cost.get(state["hotel_type"], 1000)
    transport = transport_cost.get(state["transport"], 1500)

    total_cost = (place_cost + hotel) * state["days"] + transport

    state["estimated_cost"] = total_cost

    return state


# =========================
# CONDITION NODE
# =========================

def check_budget(state: TravelState):
    if state["budget"] >= state["estimated_cost"]:
        return "budget_ok"
    else:
        return "budget_low"


# =========================
# NODE 3: GENERATE NORMAL PLAN
# =========================

def generate_travel_plan(state: TravelState):
    state["status"] = "Budget is enough"

    prompt = f"""
You are an AI Travel Agent.
Create a helpful travel plan in simple Hinglish.

Destination: {state['destination']}
Days: {state['days']}
Budget: {state['budget']}
Estimated Cost: {state['estimated_cost']}
Hotel Type: {state['hotel_type']}
Transport: {state['transport']}

Give trip summary, day wise plan, food suggestion, travel tips and final advice.
"""

    ai_answer = generate_ai_answer(prompt)

    if is_bad_ai_answer(ai_answer):
        day_plan = create_day_plan(state["destination"], state["days"])

        ai_answer = f"""
✅ **Travel Plan Ready**

Aapka budget is trip ke liye enough hai. Aap comfortably ye trip enjoy kar sakte ho.

### Trip Details

**Destination:** {state['destination']}  
**Days:** {state['days']}  
**Your Budget:** ₹{state['budget']}  
**Estimated Cost:** ₹{state['estimated_cost']}  
**Hotel Type:** {state['hotel_type']}  
**Transport:** {state['transport']}  

### Status

✅ Budget enough hai.

### Day Wise Plan

{day_plan}

### Food Suggestion

- Local food try karo.
- Budget restaurants choose karo.
- Tourist area ke expensive cafes avoid karo.
- Water bottle aur snacks carry karo.

### Travel Tips

- Hotel advance me book karo.
- Local transport ka use karo.
- Emergency ke liye extra cash rakho.
- ID proof aur booking details phone me save rakho.

### Final Advice

Aapka budget **₹{state['budget']}** hai aur estimated cost **₹{state['estimated_cost']}** hai, isliye ye trip possible hai.
"""

    state["final_answer"] = ai_answer
    return state


# =========================
# NODE 4: GENERATE LOW BUDGET PLAN
# =========================

def generate_low_budget_plan(state: TravelState):
    state["status"] = "Budget is low"

    extra_needed = state["estimated_cost"] - state["budget"]

    prompt = f"""
You are an AI Travel Agent.
User budget is low.
Create budget travel suggestions in simple Hinglish.

Destination: {state['destination']}
Days: {state['days']}
Budget: {state['budget']}
Estimated Cost: {state['estimated_cost']}
Extra Needed: {extra_needed}
Hotel Type: {state['hotel_type']}
Transport: {state['transport']}

Explain why budget is low, how to reduce cost, better low budget plan and final suggestion.
"""

    ai_answer = generate_ai_answer(prompt)

    if is_bad_ai_answer(ai_answer):
        better_days = max(2, min(3, state["days"]))
        low_budget_plan = create_day_plan(state["destination"], better_days)

        ai_answer = f"""
⚠️ **Budget Low Hai**

Aapka **{state['destination']}** trip ka estimated cost **₹{state['estimated_cost']}** aa raha hai, lekin aapka budget **₹{state['budget']}** hai.  
Is trip ke liye approx **₹{extra_needed} extra** budget chahiye.

### Trip Details

**Destination:** {state['destination']}  
**Days:** {state['days']}  
**Your Budget:** ₹{state['budget']}  
**Estimated Cost:** ₹{state['estimated_cost']}  
**Extra Needed:** ₹{extra_needed}  
**Hotel Type:** {state['hotel_type']}  
**Transport:** {state['transport']}  

### Budget Low Kyu Hai?

{state['days']} days ke liye hotel, food, local travel, transport aur sightseeing ka total cost aapke budget se zyada ho raha hai.

### Cost Reduce Karne Ke Tarike

- Trip ko **{state['days']} days se {better_days} days** karo.
- Budget hotel ya hostel choose karo.
- Cab/Flight ki jagah bus ya train use karo.
- Local sightseeing ke liye shared taxi use karo.
- Expensive cafes aur restaurants avoid karo.
- Off-season me travel karo.
- Paid activities kam rakho.

### Better Low Budget Plan

{low_budget_plan}

### Final Suggestion

Aapke **₹{state['budget']}** budget ke andar **{better_days} days {state['destination']} trip** better rahega.  
Agar aapko **{state['days']} days trip** hi karni hai, to approx **₹{state['estimated_cost']}** budget rakhna better hoga.
"""

    state["final_answer"] = ai_answer
    return state


# =========================
# LANGGRAPH WORKFLOW
# =========================

graph = StateGraph(TravelState)

graph.add_node("extract_details", extract_details)
graph.add_node("calculate_cost", calculate_cost)
graph.add_node("generate_travel_plan", generate_travel_plan)
graph.add_node("generate_low_budget_plan", generate_low_budget_plan)

graph.set_entry_point("extract_details")

graph.add_edge("extract_details", "calculate_cost")

graph.add_conditional_edges(
    "calculate_cost",
    check_budget,
    {
        "budget_ok": "generate_travel_plan",
        "budget_low": "generate_low_budget_plan"
    }
)

graph.add_edge("generate_travel_plan", END)
graph.add_edge("generate_low_budget_plan", END)

travel_app = graph.compile()


# =========================
# STREAMLIT UI
# =========================

st.subheader("Ask Your Travel Question")

user_question = st.text_area(
    "Enter your question",
    placeholder="Example: Plan a Manali trip for 5 days under 10000 by bus with budget hotel"
)

if st.button("Ask AI Travel Agent"):
    if user_question.strip() == "":
        st.warning("Please enter your travel question.")
    else:
        input_state = {
            "user_question": user_question,
            "destination": "",
            "days": 0,
            "budget": 0,
            "hotel_type": "",
            "transport": "",
            "estimated_cost": 0,
            "status": "",
            "final_answer": ""
        }

        result = travel_app.invoke(input_state)

        st.success("AI Travel Agent Response")

        st.subheader("Extracted Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Destination:**", result["destination"])
            st.write("**Days:**", result["days"])
            st.write("**Budget:** ₹", result["budget"])

        with col2:
            st.write("**Hotel Type:**", result["hotel_type"])
            st.write("**Transport:**", result["transport"])
            st.write("**Estimated Cost:** ₹", result["estimated_cost"])

        if result["status"] == "Budget is enough":
            st.success(result["status"])
        else:
            st.error(result["status"])

        st.subheader("AI Answer")
        st.markdown(result["final_answer"])

        st.subheader("LangGraph Flow")
        st.code("""
START
  ↓
extract_details
  ↓
calculate_cost
  ↓
check_budget
  ├── budget_ok  → generate_travel_plan
  └── budget_low → generate_low_budget_plan
  ↓
END
""")