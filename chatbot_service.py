import requests
import google.generativeai as genai
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple

# Configuration
# Weaviate API Configuration
USE_LOCAL_WEAVIATE = False  # Changed to False to use production
LOCAL_WEAVIATE_URL = "http://127.0.0.1:8001"  # Local testing URL
PRODUCTION_WEAVIATE_URL = "https://priceless-weaviate-production.up.railway.app"  # Production URL

WEAVIATE_API_URL = LOCAL_WEAVIATE_URL if USE_LOCAL_WEAVIATE else PRODUCTION_WEAVIATE_URL

print(f"🔧 Using Weaviate API: {WEAVIATE_API_URL}")

genai.configure(api_key="AIzaSyABQWccDN9IN329aW39pKblYKFX2E-2D8I")

# In-memory storage for summaries
chat_summaries = {}

def should_answer_question(user_query: str, conversation_context: str = "") -> bool:
    """
    Step 1: Determine if we should answer this question at all
    Uses LLM for accurate classification
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    You are a helpful assistant for a Turkish grocery shopping app.
    
    Determine if the following question is related to:
    - Food products, groceries, or shopping
    - Market chains, prices, or product comparisons
    - Cooking, recipes, or food preparation
    
    Context from previous conversation:
    {conversation_context}
    
    User question: "{user_query}"
    
    Answer with only YES if it's related to food/shopping/markets, or NO if it's completely off-topic.
    """
    
    try:
        response = model.generate_content(prompt)
        return "YES" in response.text.upper()
    except Exception as e:
        print(f"Error in should_answer_question: {e}")
        return True

def needs_product_search(user_query: str, conversation_context: str = "") -> bool:
    """
    Step 2: Determine if we need to search for products or can answer directly
    Uses LLM for accurate decision making
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    You are a classification assistant for a Turkish shopping app.
    
    Determine if this question needs product search (prices, availability, comparison) 
    or can be answered with general knowledge (cooking tips, nutrition, etc.).
    
    Context: {conversation_context}
    Question: "{user_query}"
    
    Examples:
    - "elma fiyatı nedir?" → YES (needs search)
    - "elma nasıl saklanır?" → NO (general knowledge)
    - "bu ürünler ne kadar?" → YES (needs search)
    - "bu malzemeyi nasıl kullanırım?" → NO (general knowledge)
    
    Answer with only YES or NO.
    """
    
    try:
        response = model.generate_content(prompt)
        return "YES" in response.text.upper()
    except Exception as e:
        print(f"Error in needs_product_search: {e}")
        return True

def extract_search_terms(user_query: str, conversation_context: str = "") -> List[str]:
    """
    Step 3: Extract product names/terms that need to be searched
    Uses LLM for accurate extraction with better prompting
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    Extract product names from this Turkish query and return ONLY a JSON array.
    
    Query: "{user_query}"
    Context: {conversation_context}
    
    Rules:
    - Extract specific food/product names (muz, elma, süt, etc.)
    - Use base product names, not adjectives
    - For follow-up questions with "bu/bunlar", check context
    
    Examples:
    Query: "muz fiyatları ne kadar?" → ["muz"]
    Query: "süt ve peynir ne kadar?" → ["süt", "peynir"]  
    Query: "market nasıl?" → []
    
    IMPORTANT: Return ONLY the JSON array, no other text:
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        if '[' in response_text and ']' in response_text:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            json_part = response_text[start:end]
        else:
            json_part = response_text
        
        extracted = json.loads(json_part)
        if isinstance(extracted, list) and extracted:
            print(f"Successfully extracted terms: {extracted}")
            return extracted
        else:
            print(f"Empty extraction result, falling back to heuristic")
            return extract_terms_heuristic(user_query, conversation_context)
            
    except Exception as e:
        print(f"Error in extract_search_terms: {e}")
        print(f"Response was: {response.text if 'response' in locals() else 'No response'}")
        return extract_terms_heuristic(user_query, conversation_context)

def extract_terms_heuristic(user_query: str, conversation_context: str = "") -> List[str]:
    """
    Fallback heuristic method to extract product terms
    """
    product_keywords = [
        'elma', 'muz', 'süt', 'ekmek', 'tavuk', 'et', 'sebze', 'meyve',
        'domates', 'salatalık', 'patates', 'soğan', 'biber', 'havuç',
        'peynir', 'yoğurt', 'tereyağ', 'makarna', 'pirinç', 'bulgur',
        'çay', 'kahve', 'şeker', 'tuz', 'yağ', 'un', 'balık', 'kıyma',
        'fasulye', 'nohut', 'mercimek', 'pilic', 'dana', 'kuzu'
    ]
    
    query_lower = user_query.lower()
    found_products = []
    
    for product in product_keywords:
        if product in query_lower:
            found_products.append(product)
    
    if any(word in query_lower for word in ['bu', 'bunlar', 'şu', 'o']) and conversation_context:
        context_lower = conversation_context.lower()
        for product in product_keywords:
            if product in context_lower:
                found_products.append(product)
    
    unique_products = list(set(found_products))
    
    if unique_products:
        print(f"Heuristic extraction found: {unique_products}")
        return unique_products
    else:
        print("No products found, using fallback search term")
        if any(word in query_lower for word in ['fiyat', 'ne kadar', 'kaç para', 'ürün']):
            return ['meyve']
        return []

def search_products_api(search_term: str, top_k: int = 20) -> List[Dict]:
    """
    Step 4: Use Weaviate semantic search API only
    """
    print(f"🔍 Searching Weaviate for: '{search_term}' (limit: {top_k})")
    results = search_products_weaviate(search_term, limit=top_k)
    
    if results:
        print(f"✅ Found {len(results)} products from Weaviate")
        return results
    else:
        print("❌ No results from Weaviate")
        return []

def llm_filter_and_score_products(user_query: str, products: List[Dict], conversation_context: str = "") -> List[Dict]:
    """
    Step 5: Use LLM to intelligently filter, score and rank products
    This replaces all manual regex logic with AI intelligence
    """
    if not products:
        return []
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Prepare product data for LLM analysis
    product_summaries = []
    for i, product in enumerate(products):
        summary = f"{i}: {product['name']} | {product['price']} TL | {product['market_name']}"
        if product.get('main_category'):
            summary += f" | Category: {product.get('main_category')}"
        product_summaries.append(summary)
    
    products_text = "\n".join(product_summaries)
    
    prompt = f"""
    You are a helpful shopping assistant helping a Turkish user find products.
    
    User said: "{user_query}"
    Previous conversation: {conversation_context}
    
    Here are the available products:
    {products_text}
    
    Your job: Help the user by selecting the most relevant products for their needs.
    
    Think about what the user really wants:
    - If they mention "diğer marketler" (other markets), they want alternatives to what they mentioned
    - If they ask for "elma" (apple), they probably want actual apples, not apple juice or vinegar
    - If they mention a specific store, understand whether they want only that store or are excluding it
    - If they ask for prices, they want to see different options to compare
    - Be helpful and flexible - don't be overly strict about exact wording
    
    Select products that would genuinely help this user. Score each selected product:
    - 10: Perfect match for what they're asking
    - 8-9: Very good option they'd probably want
    - 7: Good alternative option
    - 6: Somewhat relevant, might be useful
    
    Return a JSON array with your selections:
    [
        {{"index": 0, "score": 9, "reason": "fresh apple from alternative market"}},
        {{"index": 3, "score": 8, "reason": "another apple variety they might like"}},
        {{"index": 7, "score": 7, "reason": "good price alternative"}}
    ]
    
    Be helpful and inclusive rather than restrictive. The user wants good options.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        if '[' in response_text and ']' in response_text:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            json_part = response_text[start:end]
        else:
            json_part = response_text
        
        scoring_results = json.loads(json_part)
        
        # Sort by score and extract relevant products
        scoring_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        relevant_products = []
        for result in scoring_results:
            idx = result.get('index')
            score = result.get('score', 0)
            if 0 <= idx < len(products) and score >= 6:  # Only include good matches
                relevant_products.append(products[idx])
        
        print(f"LLM filtered to {len(relevant_products)} relevant products")
        return relevant_products[:15]  # More generous with results
        
    except Exception as e:
        print(f"Error in LLM filtering: {e}")
        print(f"Response was: {response.text if 'response' in locals() else 'No response'}")
        # More generous fallback - include more products
        return products[:12]

def llm_organize_for_response(user_query: str, products: List[Dict], conversation_context: str = "") -> Dict:
    """
    Step 6: Use LLM to organize products for optimal response generation
    """
    if not products:
        return {"primary": [], "secondary": [], "response_type": "no_results"}
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Prepare product data
    product_summaries = []
    for i, product in enumerate(products):
        summary = f"{i}: {product['name']} | {product['price']} TL | {product['market_name']}"
        product_summaries.append(summary)
    
    products_text = "\n".join(product_summaries)
    
    prompt = f"""
    You're helping organize a response for a Turkish shopping query.
    
    User asked: "{user_query}"
    Context: {conversation_context}
    
    Available products to include in response:
    {products_text}
    
    Think about how to best help this user:
    - What's the main thing they want to know?
    - How should we present these products to be most helpful?
    - Should we focus on cheapest options, variety, specific markets, or comparison?
    
    Organize the products to create the best possible answer:
    - Primary products: The main ones to highlight (3-8 products)
    - Secondary products: Additional options if helpful (0-3 products)
    
    What type of response would be most helpful?
    - "price_comparison": Show different price options
    - "market_alternatives": Show options from different markets  
    - "product_variety": Show different types/brands
    - "simple_answer": Just show the best few options
    
    Return JSON:
    {{
        "response_type": "price_comparison" | "market_alternatives" | "product_variety" | "simple_answer",
        "primary_products": [0, 1, 2, 3, 4],
        "secondary_products": [5, 6],
        "organization_strategy": "by_price" | "by_market" | "by_relevance"
    }}
    
    Select indices that will create a helpful, informative response.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"LLM organization response: {response_text}")
        
        # Extract JSON from response
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_part = response_text[start:end]
        else:
            json_part = response_text
        
        organization = json.loads(json_part)
        
        # Extract organized products with validation
        primary_indices = organization.get('primary_products', [])
        secondary_indices = organization.get('secondary_products', [])
        
        print(f"Primary indices: {primary_indices}, Secondary indices: {secondary_indices}")
        
        # Validate and filter indices
        valid_primary = [i for i in primary_indices if isinstance(i, int) and 0 <= i < len(products)]
        valid_secondary = [i for i in secondary_indices if isinstance(i, int) and 0 <= i < len(products)]
        
        organized_result = {
            "primary": [products[i] for i in valid_primary],
            "secondary": [products[i] for i in valid_secondary],
            "response_type": organization.get('response_type', 'simple_answer'),
            "strategy": organization.get('organization_strategy', 'by_relevance')
        }
        
        # Generous fallback: always provide helpful products
        if not organized_result['primary']:
            print("No primary products selected, using generous fallback")
            organized_result['primary'] = products[:min(6, len(products))]
            organized_result['response_type'] = 'simple_answer'
            organized_result['strategy'] = 'by_relevance'
        
        print(f"LLM organized: {len(organized_result['primary'])} primary, {len(organized_result['secondary'])} secondary")
        return organized_result
        
    except Exception as e:
        print(f"Error in LLM organization: {e}")
        print(f"Response was: {response.text if 'response' in locals() else 'No response'}")
        # Generous fallback organization
        return {
            "primary": products[:min(6, len(products))],
            "secondary": products[6:min(9, len(products))] if len(products) > 6 else [],
            "response_type": "simple_answer",
            "strategy": "by_relevance"
        }

def generate_intelligent_response(user_query: str, organized_products: Dict, conversation_context: str = "") -> str:
    """
    Step 7: Generate intelligent response based on organized products
    """
    primary_products = organized_products.get('primary', [])
    secondary_products = organized_products.get('secondary', [])
    response_type = organized_products.get('response_type', 'simple_answer')
    
    if not primary_products:
        return "Üzgünüm, aradığınız ürünle ilgili bilgi bulamadım."
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Format products for response
    primary_text = ""
    for product in primary_products:
        market = product.get('market_name', 'bilinmeyen market')
        primary_text += f"* **{product['name']}** - {market} - {product['price']} TL\n"
        if product.get('product_link'):
            primary_text += f"[Ürüne git]({product['product_link']})\n"
        primary_text += "\n"
    
    secondary_text = ""
    for product in secondary_products[:3]:  # Limit secondary products
        market = product.get('market_name', 'bilinmeyen market')
        secondary_text += f"* **{product['name']}** - {market} - {product['price']} TL\n"
        if product.get('product_link'):
            secondary_text += f"[Ürüne git]({product['product_link']})\n"
        secondary_text += "\n"
    
    prompt = f"""
    You're a helpful Turkish shopping assistant creating a response.
    
    User asked: "{user_query}"
    Previous chat: {conversation_context}
    Response type: {response_type}
    
    Main products to mention:
    {primary_text}
    
    Additional options (if relevant):
    {secondary_text}
    
    Create a natural, helpful response in Turkish that:
    1. Directly addresses what the user asked
    2. Includes the market name for each product
    3. Presents prices clearly
    4. Feels conversational and friendly
    5. Uses the exact product information provided (don't modify names/prices)
    6. Preserves the [Ürüne git] links
    7. IMPORTANT: Keep the product format exactly as provided with ** around names
    
    If they mentioned excluding a store, acknowledge that and focus on alternatives.
    If they want price comparison, organize by price.
    If they want market alternatives, group by markets.
    
    Write a complete, helpful response in Turkish, preserving all product details exactly as provided.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating intelligent response: {e}")
        # Helpful fallback response
        if primary_products:
            cheapest = min(primary_products, key=lambda x: float(x['price']))
            market = cheapest.get('market_name', 'bilinmeyen market')
            return f"* **{cheapest['name']}** - {market} - {cheapest['price']} TL\n[Ürüne git]({cheapest.get('product_link', '')})"
        return "Üzgünüm, şu anda yanıt oluşturamıyorum."

def answer_general_question(user_query: str, conversation_context: str = "") -> str:
    """
    Answer general questions without product search
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    You are a helpful Turkish shopping and food assistant.
    
    Context: {conversation_context}
    User question: "{user_query}"
    
    Answer this general question about food, cooking, shopping, or markets in Turkish.
    Keep it helpful, accurate, and conversational.
    If you don't know something specific, say so politely.
    
    If the question refers to products mentioned in the context (using "bu", "bunlar", etc.),
    be specific about which products you're discussing.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error in general question: {e}")
        return "Üzgünüm, şu anda bu soruya yanıt veremiyorum."

def process_chat_message(user_query: str, conversation_context: str = "") -> str:
    """
    Main function with completely LLM-powered intelligence
    """
    print(f"Processing: {user_query}")
    
    # Step 1: Should we answer this question?
    if not should_answer_question(user_query, conversation_context):
        return "Üzgünüm, sadece yemek, market ve alışveriş ile ilgili sorularda yardımcı olabiliyorum."
    
    # Step 2: Do we need product search?
    if not needs_product_search(user_query, conversation_context):
        return answer_general_question(user_query, conversation_context)
    
    try:
        # Step 3: Extract search terms
        search_terms = extract_search_terms(user_query, conversation_context)
        print(f"Search terms: {search_terms}")
        
        # Step 4: Search for products (deduplicated)
        seen_products = set()  # Track unique products
        all_products = []
        
        for term in search_terms:
            products = search_products_api(term, top_k=20)
            for product in products:
                # Create a unique key for each product based on name and market
                product_key = f"{product['name']}_{product['market_name']}"
                if product_key not in seen_products:
                    seen_products.add(product_key)
                    all_products.append(product)
        
        print(f"Found {len(all_products)} unique products")
        
        if not all_products:
            return "Üzgünüm, aradığınız ürünlerle ilgili sonuç bulamadım."
        
        # Step 5: LLM-powered intelligent filtering and scoring
        relevant_products = llm_filter_and_score_products(user_query, all_products, conversation_context)
        
        # Step 6: LLM-powered organization for response
        organized_products = llm_organize_for_response(user_query, relevant_products, conversation_context)
        
        # Step 7: Generate intelligent response
        return generate_intelligent_response(user_query, organized_products, conversation_context)
        
    except Exception as e:
        print(f"Error in process_chat_message: {e}")
        return f"Üzgünüm, bir hata oluştu: {str(e)}"

def search_products_weaviate(search_term: str, collection: str = "SupermarketProducts3", limit: int = 20) -> List[Dict]:
    """
    Search products using Weaviate semantic search endpoint
    """
    url = f"{WEAVIATE_API_URL}/search"
    params = {
        "query": search_term,
        "collection": collection,
        "limit": limit
    }
    
    print(f"�� Searching Weaviate for '{search_term}' in collection '{collection}'")
    print(f"🔗 URL: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=30)  # Increased timeout
        
        if response.status_code == 200:
            products = response.json()
            if isinstance(products, list):
                print(f"✅ Found {len(products)} products")
                return products
            else:
                print("❌ Invalid response format")
                return []
        else:
            print(f"❌ Search failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        print("❌ Search request timed out")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Search request failed: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in search: {e}")
        return []

def get_products_from_weaviate(collection: str = "SupermarketProducts3", offset: int = 0, limit: int = 100) -> List[Dict]:
    """
    Get products from Weaviate collection using the chatbot endpoint
    """
    url = f"{WEAVIATE_API_URL}/chatbot/products"
    params = {
        "collection": collection,
        "offset": offset,
        "limit": limit
    }
    
    print(f"📦 Fetching products from collection '{collection}'")
    print(f"🔗 URL: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=30)  # Increased timeout
        
        if response.status_code == 200:
            products = response.json()
            if isinstance(products, list):
                print(f"✅ Retrieved {len(products)} products")
                return products
            else:
                print("❌ Invalid response format")
                return []
        else:
            print(f"❌ Fetch failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        print("❌ Product fetch request timed out")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Product fetch request failed: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in product fetch: {e}")
        return []

def get_available_collections() -> List[str]:
    """
    Get list of available Weaviate collections
    """
    url = f"{WEAVIATE_API_URL}/chatbot/collections"
    print(f"📚 Fetching available collections")
    print(f"🔗 URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)  # Increased timeout
        
        if response.status_code == 200:
            data = response.json()
            collections = data.get("collections", [])
            if collections:
                print(f"✅ Found collections: {collections}")
                return collections
            else:
                print("❌ No collections found")
                return ["SupermarketProducts3"]  # fallback
        else:
            print(f"❌ Collections fetch failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return ["SupermarketProducts3"]  # fallback
            
    except requests.exceptions.Timeout:
        print("❌ Collections request timed out")
        return ["SupermarketProducts3"]  # fallback
    except requests.exceptions.RequestException as e:
        print(f"❌ Collections request failed: {e}")
        return ["SupermarketProducts3"]  # fallback
    except Exception as e:
        print(f"❌ Unexpected error in collections fetch: {e}")
        return ["SupermarketProducts3"]  # fallback

def get_product_knowledge_base(collection: str = "SupermarketProducts3", limit: int = 500) -> List[Dict]:
    """
    Get a subset of products from Weaviate for RAG knowledge base
    Used to provide context to the LLM about available products
    """
    try:
        # Get products in batches to avoid overwhelming the API
        all_products = []
        batch_size = 100
        offset = 0
        
        while len(all_products) < limit:
            batch = get_products_from_weaviate(
                collection=collection, 
                offset=offset, 
                limit=min(batch_size, limit - len(all_products))
            )
            
            if not batch:  # No more products
                break
                
            all_products.extend(batch)
            offset += batch_size
            
            if len(batch) < batch_size:  # Last batch
                break
        
        print(f"Retrieved {len(all_products)} products for knowledge base")
        return all_products
        
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        return []

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

def process_conversation_history(messages: List[Dict], user_id: str) -> Tuple[str, List[Dict]]:
    """
    Process conversation history and return context string and updated messages.
    Returns: (context_string, updated_messages)
    """
    WINDOW_SIZE = 5  # Number of recent messages to keep in full
    MAX_MESSAGES = 15  # When to start summarizing
    
    if len(messages) <= WINDOW_SIZE:
        # If we have few messages, just return them all
        context = "\n".join(f"{msg['role'].upper()}: {msg['content']}" 
                          for msg in messages)
        return context, messages
    
    # If we have more than MAX_MESSAGES, summarize older ones
    if len(messages) > MAX_MESSAGES:
        older_messages = messages[:-10]  # Messages to summarize
        recent_messages = messages[-10:]  # Keep last 10 messages as is
        
        # Create or update summary
        chat_summaries[user_id] = create_conversation_summary(older_messages, user_id)
        
        # Return recent messages only
        messages = recent_messages
    
    # Get any existing summary
    summary = chat_summaries.get(user_id, "")
    
    # Format context with summary and recent messages
    recent_context = "\n".join(f"{msg['role'].upper()}: {msg['content']}" 
                             for msg in messages[-WINDOW_SIZE:])
    
    context = f"ÖZET: {summary}\n\nSON MESAJLAR:\n{recent_context}" if summary else recent_context
    
    return context, messages

def enhanced_product_search_with_rag(user_query: str, conversation_history: List[Dict], user_id: str) -> str:
    """
    Enhanced version that uses both semantic search and knowledge base for better results
    Now accepts conversation_history as a list of message dictionaries
    """
    print(f"Enhanced RAG search for: {user_query}")
    
    # Process conversation history and get context
    context, updated_history = process_conversation_history(conversation_history, user_id)
    
    # Step 1: Should we answer this question?
    if not should_answer_question(user_query, context):
        return "Üzgünüm, sadece yemek, market ve alışveriş ile ilgili sorularda yardımcı olabiliyorum."
    
    # Step 2: Do we need product search?
    if not needs_product_search(user_query, context):
        return answer_general_question(user_query, context)
    
    # Step 3: Extract search terms
    search_terms = extract_search_terms(user_query, context)
    print(f"Search terms: {search_terms}")
    
    # Step 4: Get both search results and knowledge base
    search_results = []
    for term in search_terms:
        results = search_products_weaviate(term, limit=20)
        search_results.extend(results)
    
    print(f"Found {len(search_results)} total products from search")
    
    # Step 5: If search results are limited, supplement with knowledge base
    if len(search_results) < 10:
        print("Supplementing with knowledge base...")
        knowledge_base = get_product_knowledge_base(limit=200)
        
        # Filter knowledge base by search terms
        for product in knowledge_base:
            product_name = product.get('name', '').lower()
            if any(term.lower() in product_name for term in search_terms):
                search_results.append(product)
    
    # Step 6: LLM filtering and organization
    relevant_products = llm_filter_and_score_products(user_query, search_results, context)
    organized_products = llm_organize_for_response(user_query, relevant_products, context)
    
    # Step 7: Generate response
    return generate_intelligent_response(user_query, organized_products, context) 