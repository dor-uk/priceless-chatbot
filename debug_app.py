import streamlit as st
import json
from chatbot_service import (
    should_answer_question,
    needs_product_search,
    extract_search_terms,
    search_products_api,
    llm_filter_and_score_products,
    llm_organize_for_response,
    generate_intelligent_response,
    answer_general_question,
    process_chat_message
)

st.set_page_config(page_title="RAG Chatbot Debugger", layout="wide")
st.title("🔍 RAG Chatbot Pipeline Debugger")

# Sidebar for configuration
st.sidebar.header("Configuration")
debug_mode = st.sidebar.checkbox("Debug Mode (Step by Step)", value=True)
show_raw_data = st.sidebar.checkbox("Show Raw JSON Data", value=False)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    user_query = st.text_input("User Query:", value="muz fiyatları ne kadar?")
    conversation_context = st.text_area("Conversation Context:", value="", height=100)
    
    if st.button("🚀 Process Query", type="primary"):
        if user_query:
            st.session_state.query = user_query
            st.session_state.context = conversation_context
            st.session_state.process = True

with col2:
    st.header("Quick Test Queries")
    if st.button("🍌 Banana prices"):
        st.session_state.query = "muz fiyatları ne kadar?"
        st.session_state.context = ""
        st.session_state.process = True
    
    if st.button("🥛 Milk products"):
        st.session_state.query = "süt ürünleri nerede ucuz?"
        st.session_state.context = ""
        st.session_state.process = True
        
    if st.button("🍅 Market comparison"):
        st.session_state.query = "en ucuz domates hangi markette?"
        st.session_state.context = ""
        st.session_state.process = True
    
    if st.button("❓ Follow-up question"):
        st.session_state.query = "en ucuz olanı hangisi?"
        st.session_state.context = "USER: muz fiyatları ne kadar?\nASSISTANT: Muz Yerli: 68.97 TL (Mopas), Muz İthal: 91.97 TL (Mopas)"
        st.session_state.process = True
    
    if st.button("🍳 Cooking question"):
        st.session_state.query = "muz nasıl saklanır?"
        st.session_state.context = ""
        st.session_state.process = True

# Processing section
if hasattr(st.session_state, 'process') and st.session_state.process:
    query = st.session_state.query
    context = st.session_state.context
    
    st.header("🔍 Pipeline Execution")
    
    if debug_mode:
        # Step-by-step debugging
        st.subheader("Step 1: Should Answer Question?")
        with st.expander("🎯 Topic Classification", expanded=True):
            try:
                should_answer = should_answer_question(query, context)
                st.write(f"**Result:** {'✅ YES' if should_answer else '❌ NO'}")
                st.write(f"**Query:** {query}")
                st.write(f"**Context:** {context}")
                
                if not should_answer:
                    st.error("Query rejected as off-topic. Pipeline stops here.")
                    st.stop()
                else:
                    st.success("Query accepted. Proceeding to next step.")
            except Exception as e:
                st.error(f"Error in step 1: {e}")
                st.stop()
        
        st.subheader("Step 2: Needs Product Search?")
        with st.expander("🔍 Search Decision", expanded=True):
            try:
                needs_search = needs_product_search(query, context)
                st.write(f"**Result:** {'🔍 SEARCH' if needs_search else '💭 GENERAL'}")
                st.write(f"**Interpretation:** {'Product search required' if needs_search else 'Can answer with general knowledge'}")
                
                if not needs_search:
                    st.info("No product search needed. Getting general answer...")
                    general_answer = answer_general_question(query, context)
                    st.subheader("Final Answer (General)")
                    st.write(general_answer)
                    st.stop()
                else:
                    st.success("Product search required. Continuing pipeline.")
            except Exception as e:
                st.error(f"Error in step 2: {e}")
                st.stop()
        
        st.subheader("Step 3: Extract Search Terms")
        with st.expander("🏷️ Term Extraction", expanded=True):
            try:
                search_terms = extract_search_terms(query, context)
                st.write(f"**Extracted Terms:** {search_terms}")
                if show_raw_data:
                    st.json(search_terms)
            except Exception as e:
                st.error(f"Error in step 3: {e}")
                st.stop()
        
        st.subheader("Step 4: Search Products")
        with st.expander("🛒 API Search Results", expanded=True):
            try:
                all_products = []
                for term in search_terms:
                    st.write(f"**Searching for:** {term}")
                    products = search_products_api(term, top_k=20)
                    st.write(f"Found {len(products)} products for '{term}'")
                    all_products.extend(products)
                
                st.write(f"**Total products:** {len(all_products)}")
                
                if show_raw_data:
                    st.json(all_products[:5])  # Show first 5 for readability
                
                # Show product summary
                if all_products:
                    st.write("**Product Summary:**")
                    for i, product in enumerate(all_products[:10]):
                        st.write(f"{i}: {product['name']} - {product['price']} TL - {product['market_name']}")
            except Exception as e:
                st.error(f"Error in step 4: {e}")
                st.stop()
        
        st.subheader("Step 5: LLM Product Filtering & Scoring")
        with st.expander("🧠 AI-Powered Filtering", expanded=True):
            try:
                relevant_products = llm_filter_and_score_products(query, all_products, context)
                st.write(f"**LLM filtered to:** {len(relevant_products)} relevant products")
                
                st.write("**AI-Selected Products:**")
                for i, product in enumerate(relevant_products):
                    categories = f"{product.get('main_category', '')}/{product.get('sub_category', '')}/{product.get('lowest_category', '')}"
                    st.write(f"🧠 {i+1}: {product['name']} - {product['price']} TL - {product['market_name']} - {categories}")
                
                if show_raw_data:
                    st.json(relevant_products)
            except Exception as e:
                st.error(f"Error in step 5: {e}")
                st.stop()
        
        st.subheader("Step 6: LLM Organization for Response")
        with st.expander("📊 AI-Powered Organization", expanded=True):
            try:
                organized_products = llm_organize_for_response(query, relevant_products, context)
                st.write(f"**Response Type:** {organized_products.get('response_type', 'unknown')}")
                st.write(f"**Organization Strategy:** {organized_products.get('strategy', 'unknown')}")
                st.write(f"**Primary Products:** {len(organized_products.get('primary', []))}")
                st.write(f"**Secondary Products:** {len(organized_products.get('secondary', []))}")
                
                st.write("**Primary Products (Main Focus):**")
                for i, product in enumerate(organized_products.get('primary', [])):
                    st.write(f"🏆 {i+1}: {product['name']} - {product['price']} TL - {product['market_name']}")
                
                st.write("**Secondary Products (Additional Options):**")
                for i, product in enumerate(organized_products.get('secondary', [])):
                    st.write(f"➕ {i+1}: {product['name']} - {product['price']} TL - {product['market_name']}")
                
                if show_raw_data:
                    st.json(organized_products)
            except Exception as e:
                st.error(f"Error in step 6: {e}")
                st.stop()
        
        st.subheader("Step 7: Generate Intelligent Response")
        with st.expander("💬 AI Response Generation", expanded=True):
            try:
                final_response = generate_intelligent_response(query, organized_products, context)
                st.write("**AI-Generated Response:**")
                st.markdown(final_response)
            except Exception as e:
                st.error(f"Error in step 7: {e}")
                st.stop()
    
    else:
        # Direct processing (non-debug mode)
        st.subheader("🚀 Direct Processing")
        try:
            with st.spinner("Processing query..."):
                response = process_chat_message(query, context)
            
            st.subheader("Final Response")
            st.markdown(response)
        except Exception as e:
            st.error(f"Error in direct processing: {e}")

    # Reset process flag
    st.session_state.process = False

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Debug Info")
st.sidebar.write("Use this interface to debug each step of the RAG pipeline and identify quality issues.")

if st.sidebar.button("🔄 Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun() 