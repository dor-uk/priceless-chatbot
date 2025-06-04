#!/usr/bin/env python3
"""
Test script for Weaviate integration with chatbot service
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot_service import (
    get_available_collections,
    search_products_weaviate, 
    get_product_knowledge_base,
    enhanced_product_search_with_rag
)

def test_collections():
    """Test getting available collections"""
    print("🔍 Testing collections endpoint...")
    collections = get_available_collections()
    print(f"Available collections: {collections}")
    return len(collections) > 0

def test_weaviate_search():
    """Test Weaviate semantic search"""
    print("\n🔍 Testing Weaviate semantic search...")
    results = search_products_weaviate("elma", limit=5)
    print(f"Found {len(results)} products for 'elma':")
    for i, product in enumerate(results[:3]):
        print(f"  {i+1}: {product.get('name')} - {product.get('price')} TL - {product.get('market_name')}")
    return len(results) > 0

def test_knowledge_base():
    """Test knowledge base retrieval"""
    print("\n🔍 Testing knowledge base retrieval...")
    products = get_product_knowledge_base(limit=10)
    print(f"Retrieved {len(products)} products for knowledge base:")
    for i, product in enumerate(products[:3]):
        print(f"  {i+1}: {product.get('name')} - {product.get('price')} TL - {product.get('market_name')}")
    return len(products) > 0

def test_enhanced_rag():
    """Test enhanced RAG functionality"""
    print("\n🔍 Testing enhanced RAG chatbot...")
    test_queries = [
        "elma fiyatları nedir?",
        "en ucuz süt hangisi?",
        "meyve çeşitleri neler var?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = enhanced_product_search_with_rag(query)
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            print("✅ Success")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Weaviate Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Collections", test_collections),
        ("Weaviate Search", test_weaviate_search),
        ("Knowledge Base", test_knowledge_base),
        ("Enhanced RAG", test_enhanced_rag)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"\n❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, result in results:
        print(f"  {'✅' if result else '❌'} {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Weaviate integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main() 