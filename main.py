from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chatbot_service import process_chat_message, enhanced_product_search_with_rag, get_available_collections, get_product_knowledge_base
import google.generativeai as genai
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for conversation context
user_conversations = {}
user_summaries = {}

def create_conversation_summary(messages: List[Dict], user_id: str) -> str:
    """Create a summary of conversation messages to preserve context while reducing tokens."""
    if not messages:
        return ""
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Convert messages to text format
    conversation_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)
    
    prompt = f"""
    You are helping to summarize a conversation between a user and a Turkish shopping assistant.
    
    Please create a concise summary of the following conversation that preserves:
    - Product names or categories the user has asked about
    - Any preferences they've expressed (price ranges, stores, etc.)
    - Important context that might be relevant for future questions
    
    Conversation to summarize:
    {conversation_text}
    
    Create a brief summary in Turkish that captures the essential context. Keep it under 100 words.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        return f"Kullanıcı {len(messages)} mesajlık bir konuşma yaptı."

def get_conversation_context(user_id: str, window_size: int = 5) -> str:
    """Get conversation context with summary of older messages."""
    if user_id not in user_conversations:
        return ""
        
    conversation_history = user_conversations[user_id]
    
    if len(conversation_history) <= window_size:
        # If we have few messages, just return them all
        return "\n".join(f"{msg['role'].upper()}: {msg['content']}" 
                        for msg in conversation_history)
    
    # Get the summary of older messages
    older_messages = conversation_history[:-window_size]
    recent_messages = conversation_history[-window_size:]
    
    # Create or get summary
    if user_id not in user_summaries:
        user_summaries[user_id] = create_conversation_summary(older_messages, user_id)
    
    # Combine summary with recent messages
    summary = user_summaries[user_id]
    recent_context = "\n".join(f"{msg['role'].upper()}: {msg['content']}" 
                              for msg in recent_messages)
    
    return f"ÖZET: {summary}\n\nSON MESAJLAR:\n{recent_context}"

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/")
def root():
    return {"message": "Chatbot API is running with RAG approach."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_input = request.message
    user_id = request.user_id

    try:
        # Get conversation context
        if user_id not in user_conversations:
            user_conversations[user_id] = []
        
        conversation_history = user_conversations[user_id]
        
        # Create context string from recent messages (last 6 messages)
        context = ""
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for msg in recent_history:
            context += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Process the message using new RAG approach
        response = process_chat_message(user_input, context)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Keep only last 20 messages to prevent memory bloat
        if len(conversation_history) > 20:
            user_conversations[user_id] = conversation_history[-20:]
        else:
            user_conversations[user_id] = conversation_history

        return JSONResponse(content={"response": response}, media_type="application/json; charset=utf-8")
    
    except Exception as e:
        error_response = f"Üzgünüm, bir hata oluştu: {str(e)}"
        print(f"Error in chat endpoint: {e}")
        return JSONResponse(content={"response": error_response}, media_type="application/json; charset=utf-8")

@app.post("/chat-enhanced")
def enhanced_chat_endpoint(request: ChatRequest):
    """
    Enhanced chatbot endpoint using RAG with Weaviate knowledge base
    """
    user_input = request.message
    user_id = request.user_id

    try:
        # Initialize conversation history if needed
        if user_id not in user_conversations:
            user_conversations[user_id] = []
        
        conversation_history = user_conversations[user_id]
        
        # Process using enhanced RAG approach with conversation history
        response = enhanced_product_search_with_rag(
            user_query=user_input,
            conversation_history=conversation_history,
            user_id=user_id
        )
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Store updated history
        user_conversations[user_id] = conversation_history

        return JSONResponse(
            content={"response": response},
            media_type="application/json; charset=utf-8"
        )
    
    except Exception as e:
        error_response = f"Üzgünüm, bir hata oluştu: {str(e)}"
        print(f"Error in enhanced chat endpoint: {e}")
        return JSONResponse(
            content={"response": error_response},
            media_type="application/json; charset=utf-8"
        )

@app.get("/collections")
def get_collections_endpoint():
    """
    Get available Weaviate collections
    """
    try:
        collections = get_available_collections()
        return {"collections": collections}
    except Exception as e:
        print(f"Error getting collections: {e}")
        return {"collections": [], "error": str(e)}

@app.get("/knowledge-base")
def get_knowledge_base_endpoint(collection: str = "SupermarketProducts3", limit: int = 100):
    """
    Get products from knowledge base for testing
    """
    try:
        products = get_product_knowledge_base(collection, limit)
        return {
            "collection": collection,
            "count": len(products),
            "products": products[:10] if products else [],  # Return first 10 for display
            "total_retrieved": len(products)
        }
    except Exception as e:
        print(f"Error getting knowledge base: {e}")
        return {"products": [], "error": str(e)}
    
