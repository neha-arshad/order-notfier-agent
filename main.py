import streamlit as st
import os
import asyncio
import requests
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
from agents import function_tool

load_dotenv(find_dotenv())

instance_id = os.getenv("ULTRA_INSTANCE_ID")
token = os.getenv("ULTRA_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

@function_tool
def send_whatsapp_message(recipient: str, message: str) -> str:
    try:
        url = f"https://api.ultramsg.com/{instance_id}/messages/chat"
        payload = {
            "token": token,
            "to": recipient,
            "body": message
        }
        res = requests.post(url, data=payload)
        if res.status_code == 200:
            return "📤 WhatsApp message sent successfully via UltraMsg!"
        else:
            return f"❌ Failed to send WhatsApp message: {res.text}"
    except Exception as e:
        return f"❌ Exception: {e}"

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

runConfig = RunConfig(
    model_provider=provider,
    model=model,
    tracing_disabled=True
)

agent = Agent(
    name="ordernotifier",
    instructions=(
        "Description: Notifies when a new order is placed. "
        "You are an order notification assistant. Whenever you receive an order, summarize it nicely "
        "and acknowledge it with a friendly tone. Then use the send_whatsapp_message tool "
        "to send the summary to the customer's WhatsApp number."
    ),
    tools=[send_whatsapp_message]
)


st.markdown("""
<div style="text-align:center">
    <h1 style="color:#2E7D32;">🛒 Order Notifier Agent</h1>
    <p style="font-size:18px;color:#555;">Easily notify customers of their order via <b style="color:#2E7D32;">WhatsApp</b></p>
</div>
<hr style="border:1px solid #A5D6A7;margin:10px 0;">
""", unsafe_allow_html=True)


with st.form("order_form", clear_on_submit=True):
    st.markdown("<h4 style='color:#2E7D32;'>📄 Enter Order Details</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("👤 Customer Name")
    with col2:
        customer_phone = st.text_input("📱 Customer Phone (e.g. +92XXXXXXX)")

    items = st.text_area("🛒 Order Items (e.g. 2x Laptop, 1x Mouse)")
    total = st.text_input("💵 Total Amount (e.g. 2500)")

    submitted = st.form_submit_button("✅ Submit Order")

if submitted:
    input_text = (
        f"New order placed by {customer_name}. "
        f"Items: {items}. "
        f"Total: ${total}. "
        f"Customer phone: {customer_phone}"
    )

    def run_runner_sync_in_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return Runner.run_sync(
            agent,
            input=input_text,
            run_config=runConfig
        )

    with st.spinner("⏳ Processing order and sending WhatsApp notification..."):
        result = run_runner_sync_in_loop()

    
    st.markdown(f"""
    <div style="background-color:#F1F8E9;padding:20px;border-radius:10px;border-left:5px solid #2E7D32">
        <h3 style="color:#2E7D32">✅ Order Processed Successfully!</h3>
        <h4 style="color:#33691E">📋 Agent Response:</h4>
        <p style="color:#333;font-size:16px;">{result.final_output}</p>
    </div>
    """, unsafe_allow_html=True)

