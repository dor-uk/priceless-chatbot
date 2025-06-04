# chatbot_logic.py
import pandas as pd
import google.generativeai as genai
import psycopg2
import json
from typing import Tuple
from langchain.memory import ConversationBufferMemory

# Your DB and Gemini config
DATABASE_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "small_db",
    "user": "dorukberkeyurtsizoglu",
    "password": "abcd2331"
}
MAX_SQL_RETRIES = 2
# Global in-memory storage for user chat histories
chat_memories = {}

genai.configure(api_key="AIzaSyABQWccDN9IN329aW39pKblYKFX2E-2D8I")

schema = """Table name: all_products

Columns:
- id (integer)
- main_category (text)
- sub_category (text)
- lowest_category (text)
- name (text)
- price (real)
- high_price (real)
- in_stock (text)
- product_link (text)
- page_link (text)
- image_url (text)
- date (text)
- market_name (text)"""

def translate_preserving_turkish_products(turkish_input: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    Translate the following Turkish shopping query into English, but DO NOT translate Turkish product names (like 'muz', 'biber', 'çilekli süt', etc.). 
    Just translate the rest of the sentence for clarity. the 

    Turkish Input: {turkish_input}

    English Output:"""

    response = model.generate_content(prompt)
    return response.text.strip()

# Replace st.session_state with this
def get_dummy_context():
    return ""  # Later we can make this smarter

# def get_memory_context(user_id: str = "default") -> str:
#     if user_id not in chat_memories:
#         chat_memories[user_id] = ConversationBufferMemory(return_messages=True)

#     memory = chat_memories[user_id]
#     history = memory.load_memory_variables({}).get("history", [])
#     return "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in history)

def get_memory_context(user_id: str = "default", window_size: int = 3, max_messages_before_summary: int = 4) -> str:
    if user_id not in chat_memories:
        chat_memories[user_id] = ConversationBufferMemory(return_messages=True)

    memory = chat_memories[user_id]
    history = memory.load_memory_variables({}).get("history", [])
    
    # If we have too many messages, create a summary of older ones
    if len(history) > max_messages_before_summary:
        # Keep the last few messages as-is
        recent_messages = history[-window_size * 2:]
        # Get older messages for summarization
        older_messages = history[:-window_size * 2]
        
        if older_messages:
            # Create summary of older conversation
            summary = create_conversation_summary(older_messages, user_id)
            # Return summary + recent messages
            recent_context = "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in recent_messages)
            return f"SUMMARY: {summary}\n{recent_context}" if recent_context else f"SUMMARY: {summary}"
    
    # If not too many messages, return recent ones normally
    recent = history[-window_size * 2:]  
    return "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in recent)

def create_conversation_summary(messages, user_id: str) -> str:
    """Create a summary of conversation messages to preserve context while reducing tokens."""
    if not messages:
        return ""
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Convert messages to text format
    conversation_text = "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in messages)
    
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
        # Fallback: return a simple summary
        return f"Kullanıcı {len(messages)} mesajlık geçmiş konuşma yaptı."

def save_user_message(user_id: str, message: str):
    if user_id not in chat_memories:
        chat_memories[user_id] = ConversationBufferMemory(return_messages=True)
    chat_memories[user_id].chat_memory.add_user_message(message)
    
    # Optionally clean up very old messages to prevent memory bloat
    _cleanup_old_messages(user_id, max_total_messages=20)

def save_bot_message(user_id: str, message: str):
    if user_id not in chat_memories:
        chat_memories[user_id] = ConversationBufferMemory(return_messages=True)
    chat_memories[user_id].chat_memory.add_ai_message(message)
    
    # Optionally clean up very old messages to prevent memory bloat
    _cleanup_old_messages(user_id, max_total_messages=20)

def _cleanup_old_messages(user_id: str, max_total_messages: int = 20):
    """Remove very old messages to prevent unlimited memory growth."""
    if user_id not in chat_memories:
        return
        
    memory = chat_memories[user_id]
    history = memory.load_memory_variables({}).get("history", [])
    
    if len(history) > max_total_messages:
        # Keep only the most recent messages
        recent_messages = history[-max_total_messages:]
        
        # Clear and rebuild memory with recent messages only
        memory.clear()
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                if msg.type == 'human':
                    memory.chat_memory.add_user_message(msg.content)
                elif msg.type == 'ai':
                    memory.chat_memory.add_ai_message(msg.content)

def should_use_sql(user_query: str, user_id: str = "default") -> bool:
    """
    Uses Gemini to determine if the question requires SQL/database access.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.2)
    )
    #context = get_dummy_context()
    context = get_memory_context(user_id)


    prompt = f"""
        You are a classification assistant.

        Your task is to decide whether the following user input needs to query a product database using SQL 
        or can be answered directly.

        The conversation so far:
        {context}       

        Examples:
        - "elma fiyatı nedir?" → YES
        - "A101 güvenilir mi?" → NO
        - "en ucuz yoğurt nerede?" → YES
        - "Migros ne zaman kuruldu?" → NO
        - "Mercimek çorbası için gerekli malzemeler nelerdir?" → NO
        - "bu malzemeyi yemek yaparken kullanabilir miyim?" (when context shows milk products) → NO
        - "bunlar ne kadar?" (when context shows specific products) → YES
        - "süt yemek yaparken kullanılabilir mi?" → NO
        

        Important: Look at the conversation context. If the user is asking a follow-up question about products that were already retrieved (like asking about usage, preparation, or general information about those products), answer NO. If they're asking for new product searches or price comparisons, answer YES.

        User input:
        "{user_query}"

        Answer with only YES or NO.
        """

    response = model.generate_content(prompt)
    answer = response.text.strip().upper()

    return "YES" in answer

def generate_candidate_sql(translated_query: str, table_schema: str, user_id: str = "default") -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    #context = get_dummy_context()
    context = get_memory_context(user_id)
    prompt = f"""
        You are a SQL assistant. You will not answer the user's question directly.

        Instead, your task is to generate a SQL query that fetches a **broad list of candidate products** related to the query, so that a second model can later decide which ones match exactly.

        Instructions:
        - Use a broad `LIKE` or `ILIKE` filter.
        - Do NOT optimize for price, do NOT use LIMIT, do NOT make assumptions.
        - Dont forget to write your queries case insentitive.
        - Select product name, price, market_name, and product_link.
        - Focus only on finding all products whose names loosely match what's being asked — don't solve the user's problem yet.
        - If there are measures in the user input, you need to consider them too. Those measurements are important too!

        An example sql query:
        SELECT name, price, market_name, product_link
        FROM all_products
        WHERE LOWER(name) LIKE '%elma%'
        ORDER BY price ASC

        Table schema:
        {table_schema}

        The conversation so far:
        {context}    

        User question: {translated_query}

        Now return only the SQL query (no explanation):
        """

    response = model.generate_content(prompt)
    sql = response.text.strip()

    # Optional cleanup
    if sql.startswith("```"):
        sql = sql.strip("`").replace("sql", "").strip()

    return sql

def regenerate_sql_on_error(user_query: str, failed_sql: str, error_msg: str, table_schema: str, user_id: str = "default") -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    #context = get_dummy_context()
    context = get_memory_context(user_id)
    prompt = f"""
        You are a PostgreSQL assistant.

        A SQL query was generated for a specific task. It's details are below:
            generate a SQL query that fetches a **broad list of candidate products** related to the query, 
            so that a second model can later decide which ones match exactly.

            Instructions:
            - Use a broad `LIKE` or `ILIKE` filter.
            - Do NOT optimize for price, do NOT use LIMIT, do NOT make assumptions.
            - Select product name, price, market_name, and product_link.
            - Focus only on finding all products whose names loosely match what's being asked — don't solve the user's problem yet.
            - If there are measures in the user input, you need to consider them too. Those measurements are important too!

        However the task failed. You need to find out what went wrong and try to fix it by generating an alternative sql query.
        Here are the details:
        - User question (in Turkish): "{user_query}"

        - The conversation so far:
        {context}

        - Failed SQL query:
        {failed_sql}
        - Error message:
        {error_msg}

        Your task:
        Regenerate a correct SQL SELECT query that avoids this error.
        Use only standard PostgreSQL.
        Do not include markdown, explanations, or formatting — just the corrected SQL.

        The database table is `all_products`, with this schema:
        {table_schema}

        Only output a single valid SQL query that should fix the issue.
        """

    response = model.generate_content(prompt)
    sql = response.text.strip()

    if sql.startswith("```"):
        sql = sql.strip("`").replace("sql", "").strip()

    return sql

def refine_selection_from_dataframe(user_query: str, df: pd.DataFrame, user_id: str = "default") -> pd.DataFrame:
    model = genai.GenerativeModel("gemini-2.0-flash")
    #context = get_dummy_context()
    context = get_memory_context(user_id)
    if df.empty:
        return df

    # Convert full DataFrame to readable text
    display_str = df.to_string(index=False)

    prompt = f"""
        You are a smart assistant helping with Turkish product search.

        - The user asked:
        {user_query}

        - The chat history so far:
        {context}


        - You are given a table of candidate products from a database query.
        - Your task is to analyze all the data (columns like name, category, price, market, etc.) and 
        - Decide which rows best match the user's actual intent.
        - When deciding, looking at the chat history may be helpful.
        - The ones that are semantically more similar to what user wants are more likely to be the correct products.

        Here are the candidate rows:
        {display_str}

        You have to return the row or rows that actually hold the information the user wants — no more, no less.

        Return the selected row(s) as a Python list of integers, corresponding to their row numbers in the table above 
        (zero-indexed from the top). No explanation, no markdown, just the list.

        Example:
        [0, 2]
        Note: You shouldn't assume that You need to return multiple rows always. You can return just 1 row too if the user asks for something like 'most' or 'least' or 'cheapest' etc.
        """

    response = model.generate_content(prompt)

    try:
        selected_indices = json.loads(response.text.strip())
    except:
        try:
            selected_indices = eval(response.text.strip())
        except:
            selected_indices = []

    # Safely select rows by index
    try:
        filtered_df = df.iloc[selected_indices]
    except Exception as e:
        print("Index filtering error:", e)
        filtered_df = pd.DataFrame()

    return filtered_df


def execute_sql(query: str) -> pd.DataFrame:
    conn = None
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        # Explicitly create a DataFrame from the query results
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"SQL execution error: {e}")
        # Return an empty DataFrame in case of error
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()



def safe_execute_sql_with_retry(user_query: str, initial_sql: str, schema: str, user_id: str = "default") -> Tuple[pd.DataFrame, str]:
    sql = initial_sql
    for attempt in range(1, MAX_SQL_RETRIES + 1):
        try:
            df = execute_sql(sql)
            return df, sql
        except Exception as e:
            print(f"[Retry {attempt}] SQL execution failed: {e}")
            if attempt == MAX_SQL_RETRIES:
                raise e
            print(f"Regenerating SQL query (attempt {attempt + 1})...")
            sql = regenerate_sql_on_error(user_query, sql, str(e), schema, user_id)
            print(f"New SQL:\n{sql}")
    return pd.DataFrame(), sql

def generate_human_response(user_query: str, df: pd.DataFrame, user_id: str = "default") -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    #context = get_dummy_context()
    context = get_memory_context(user_id)
    if df.empty:
        return "Üzgünüm, aradığınız ürünle tam olarak eşleşen bir sonuç bulamadım."

    table_text = df.to_string(index=False)

    prompt = f"""
        You are a helpful assistant that replies in Turkish.

        The user asked this shopping-related question in Turkish:
        "{user_query}"

        - The conversation so far:
        {context}

        Below is a list of product rows retrieved from a database. Each row includes columns like `name`, `price`, `market_name`, and possibly `product_link`.

        {table_text}

        Your task:
        - Write a natural, friendly Turkish response to the user.
        - Mention the best-matching product(s) with their name, price, and market.
        - If there's a product_link, include a "Ürüne git" phrase with the link.
        - Keep it short, relevant, and natural — like something a person would say.

        Only respond with the final message in Turkish. Do not explain your process.
        """

    response = model.generate_content(prompt)
    return response.text.strip()

def answer_without_sql(user_query: str, user_id: str = "default") -> str:
    """
    Uses Gemini to respond to general product/market/food-related questions in Turkish.
    If the question is off-topic, it refuses politely.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.5)
    )
    #context = get_dummy_context()
    context = get_memory_context(user_id)

    prompt = f"""
        You are a helpful assistant that only answers general questions about grocery shopping, 
        markets, food products, and consumer goods in Turkey.

        - The user asked the following question in Turkish:
        {user_query}

        - The chat history so far:
        {context}

        Your task:
        - Look at the conversation history to understand what products or topics were discussed previously.
        - If the user is asking a follow-up question using words like "bu" (this), "o" (that), "bunlar" (these), understand what they're referring to from the context.
        - If the question is about food, market chains (like Migros, A101), product types, shopping experience, cooking, usage of food products, etc. → respond helpfully in Turkish.
        - Be specific and refer to the products mentioned in the conversation history when answering follow-up questions.
        - If the question is unrelated (e.g., history, politics, entertainment, sports, etc.) → say that you only assist with market and product-related questions.

        Only reply in Turkish. Keep your answer short, natural, and helpful.
        """

    response = model.generate_content(prompt)
    return response.text.strip()



def rewrite_user_query(user_query: str, user_id: str = "default") -> str:
    """
    Uses Gemini to rewrite a user query by grounding it in recent conversation context.
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.3)
    )
    #context = get_dummy_context()
    context = get_memory_context(user_id)
    prompt = f"""
        You are a helpful assistant that rewrites follow-up questions about grocery products to make them clear and complete.

        Here is the recent conversation history between the user and the assistant:
        {context}

        The user just asked:
        "{user_query}"

        Your task:
        - If this is a follow-up question that refers to previous products (using words like "bu", "o", "şu", "bunlar", etc.), rewrite it to include the specific product names from the conversation history.
        - If the user asks about "this ingredient" or "these products", identify what they're referring to from the chat history.
        - Rewrite the question to be fully self-contained and unambiguous.
        - The rewritten version should include product names, market names, quantities, and other details if available in context.
        - Keep the question in Turkish.
        - DO NOT answer the question — only rewrite it.
        - Format it in a way that will be easy to convert to SQL later (e.g., direct and structured phrasing).

        Examples:
        If the conversation was about "süt" (milk) and user asks "bu malzemeyi yemek yaparken kullanabilir miyim", rewrite as: "süt yemek yaparken kullanabilir miyim"
        
        If the conversation was about specific milk products and user asks "bunlar ne kadar", rewrite as: "süt ürünleri ne kadar"

        Output only the rewritten question, nothing else.
        Important: Don't add anything to the user questions. Just paraphrase and make references explicit.
        """

    response = model.generate_content(prompt)
    return response.text.strip()

def generate_error_response(user_query: str, error_message: str) -> str:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.4)
    )

    prompt = f"""
    The user asked the following question in Turkish:

    "{user_query}"

    While trying to answer this using a product database, the following error occurred:
    "{error_message}"

    Your task:
    - Write a short, polite message in Turkish that explains something went wrong.
    - If possible, hint at what might be the cause (e.g., product name not found, invalid format, etc.).
    - Ask the user to rephrase or clarify their query.
    - Do not include technical SQL jargon.
    - Keep it friendly and natural, as if you're assisting a shopper.

    Only output the final message in Turkish.
    """

    response = model.generate_content(prompt)
    return response.text.strip()

