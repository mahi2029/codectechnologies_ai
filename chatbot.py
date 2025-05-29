import re

# Define a list of sample intents and responses
responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I help you with?"],
    "hours": ["We're open from 9 AM to 5 PM, Monday to Friday."],
    "location": ["Our store is located at 123 Main Street, Springfield."],
    "return_policy": ["You can return any item within 30 days with a receipt."],
    "goodbye": ["Thanks for chatting with us. Have a great day!"],
    "default": ["I'm sorry, I didn't understand that. Could you rephrase your question?"]
}

# Define simple patterns for each intent
patterns = {
    "greeting": r"\b(hi|hello|hey)\b",
    "hours": r"\b(hours|open|closing|time)\b",
    "location": r"\b(where|location|address|situated)\b",
    "return_policy": r"\b(return|refund|policy|exchange)\b",
    "goodbye": r"\b(bye|goodbye|see you|exit|quit)\b"
}

def match_intent(user_input):
    for intent, pattern in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return intent
    return "default"

def chatbot():
    print("Welcome to Customer Service Chatbot! Type 'quit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot:", responses["goodbye"][0])
            break
        intent = match_intent(user_input)
        print("Bot:", responses[intent][0])

if __name__ == "__main__":
    chatbot()
