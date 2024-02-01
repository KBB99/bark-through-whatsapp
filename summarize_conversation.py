from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory

llm = OpenAI(temperature=0, model="gpt-4")
user_phone_number = "whatsapp:+1234567890"

history = DynamoDBChatMessageHistory(
        table_name="SessionTable",
        session_id=user_phone_number,
    )

conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryMemory(llm=OpenAI()),
    verbose=True
)