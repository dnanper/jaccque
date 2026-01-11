"""
Memory API Test Script for SWE-Agent Use Cases.

This script tests the Memory API with code generation/debugging examples
to verify the system is working correctly before SWE-bench integration.

Run with: python test_api.py
Requires: The Memory API server running at http://localhost:8888
"""

import asyncio
import httpx
from datetime import datetime
import json


# Configuration
BASE_URL = "http://localhost:8888"
BANK_ID = "swe_agent_test"  # Test bank for SWE-Agent

# Database config (for direct embedding query - same as .env)
DATABASE_URL = "postgresql://memory:memory_secret@localhost:5432/memory_db"


async def test_view_embeddings_via_db():
    """View embeddings by querying database directly."""
    print("\n" + "=" * 60)
    print("🔢 VIEW EMBEDDINGS (Direct Database Query)")
    print("=" * 60)
    
    try:
        import asyncpg
    except ImportError:
        print("❌ asyncpg not installed. Run: pip install asyncpg")
        return False
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Query memory units with embeddings
        rows = await conn.fetch("""
            SELECT 
                id,
                text,
                fact_type,
                embedding[1:5] as embedding_preview,  -- First 5 dimensions
                array_length(embedding, 1) as embedding_dim
            FROM memory.memory_units
            WHERE bank_id = $1
            LIMIT 5
        """, BANK_ID)
        
        print(f"\nFound {len(rows)} memory units with embeddings:\n")
        
        for row in rows:
            print(f"ID: {row['id'][:8]}...")
            print(f"Text: {row['text'][:80]}...")
            print(f"Type: {row['fact_type']}")
            print(f"Embedding Dimensions: {row['embedding_dim']}")
            print(f"Embedding Preview (first 5): {list(row['embedding_preview'])}")
            print("-" * 50)
        
        await conn.close()
        print("\n✅ Embeddings retrieved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("Make sure database is accessible at:", DATABASE_URL)
        return False


async def test_health():
    """Test 1: Health check."""
    print("\n" + "=" * 60)
    print("🏥 TEST 1: Health Check")
    print("=" * 60)
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        response = await client.get("/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False


async def test_retain_code_patterns():
    """Test 2: Store code patterns and debugging knowledge."""
    print("\n" + "=" * 60)
    print("📝 TEST 2: Retain Code Patterns & Debug Knowledge")
    print("=" * 60)
    
    # Example memories that an SWE-Agent would learn
    code_memories = [
        {
            "content": "When encountering ImportError in Python, first check if the module is installed with 'pip list | grep module_name'. If not installed, use 'pip install module_name'. If installed but still errors, check PYTHONPATH or virtual environment activation.",
            "context": "error_fix",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "TypeError: 'NoneType' object is not subscriptable - This error occurs when trying to access an index or key on a None value. Fix by adding null checks: 'if result is not None: result[key]' or using 'result.get(key) if result else default'.",
            "context": "error_fix",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "In Django projects, when tests fail with database errors, ensure test database migrations are applied. Run 'python manage.py migrate --run-syncdb' before running tests. Also check if the test is using the correct database settings.",
            "context": "testing",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "For Flask applications, when debugging routing issues, use 'app.url_map' to see all registered routes. Also enable debug mode with 'app.run(debug=True)' to get detailed error pages.",
            "context": "debugging",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "When fixing bugs in Python async code, remember that 'await' can only be used inside async functions. If getting 'SyntaxError: await outside async function', wrap the code in an async function and call it with asyncio.run().",
            "context": "async_python",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "Git conflict resolution pattern: First run 'git status' to see conflicted files. Open each file and look for <<<<<<< HEAD markers. Keep the correct version, remove conflict markers, then 'git add' and 'git commit'.",
            "context": "version_control",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "When a pytest test passes locally but fails in CI, check for: 1) Different Python versions 2) Missing environment variables 3) File path differences (Windows vs Linux) 4) Database state issues 5) Timing-dependent tests.",
            "context": "ci_debugging",
            "timestamp": datetime.now().isoformat()
        },
        {
            "content": "For pandas DataFrame operations with KeyError, first check column names with 'df.columns.tolist()'. Common issues: trailing spaces in column names, case sensitivity, or column was renamed/dropped earlier in the code.",
            "context": "data_processing",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        response = await client.post(
            f"/v1/default/banks/{BANK_ID}/memories",
            json={
                "items": code_memories,
                "async": False  # Wait for completion
            }
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print(f"✅ Stored {len(code_memories)} code patterns successfully!")
            return True
        else:
            print("❌ Failed to store code patterns!")
            return False


async def test_recall_error_fixes():
    """Test 3: Recall relevant error fixes."""
    print("\n" + "=" * 60)
    print("🔍 TEST 3: Recall Error Fix Knowledge")
    print("=" * 60)
    
    queries = [
        "How to fix TypeError NoneType is not subscriptable?",
        "Python import error troubleshooting",
        "Django test database migration issues",
        "pytest fails in CI but passes locally"
    ]
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        for query in queries:
            print(f"\n🔎 Query: '{query}'")
            print("-" * 50)
            
            response = await client.post(
                f"/v1/default/banks/{BANK_ID}/memories/recall",
                json={
                    "query": query,
                    "types": ["world", "experience"],
                    "max_tokens": 2000,
                    "budget": "mid"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                print(f"Found {len(results)} relevant memories:")
                
                for i, result in enumerate(results[:3], 1):  # Show top 3
                    text = result.get("text", "")
                    context = result.get("context", "N/A")
                    print(f"\n  {i}. [{context}]")
                    print(f"     {text[:200]}..." if len(text) > 200 else f"     {text}")
            else:
                print(f"❌ Recall failed: {response.status_code}")
                print(response.text)
    
    print("\n✅ Recall test completed!")
    return True


async def test_reflect_on_problem():
    """Test 4: Use reflect to think about a coding problem."""
    print("\n" + "=" * 60)
    print("🧠 TEST 4: Reflect on a Coding Problem")
    print("=" * 60)
    
    problem = """
    I'm getting this error in my Flask application:
    
    werkzeug.exceptions.NotFound: 404 Not Found
    The requested URL was not found on the server.
    
    The route is defined as @app.route('/api/users/<int:user_id>')
    but I'm calling it with /api/users/abc
    
    What's the issue and how should I fix it?
    """
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        print(f"Problem:\n{problem}")
        print("-" * 50)
        
        response = await client.post(
            f"/v1/default/banks/{BANK_ID}/reflect",
            json={
                "query": problem,
                "budget": "mid",
                "context": "debugging a Flask web application",
                "include": {"facts": {}}  # Include facts used
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("text", "")
            based_on = data.get("based_on", [])
            
            print(f"\n💡 Analysis:\n{answer}")
            
            if based_on:
                print(f"\n📚 Based on {len(based_on)} stored memories")
            
            print("\n✅ Reflect test completed!")
            return True
        else:
            print(f"❌ Reflect failed: {response.status_code}")
            print(response.text)
            return False


async def test_bank_stats():
    """Test 5: Check bank statistics."""
    print("\n" + "=" * 60)
    print("📊 TEST 5: Bank Statistics")
    print("=" * 60)
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        response = await client.get(f"/v1/default/banks/{BANK_ID}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Bank ID: {data.get('bank_id')}")
            print(f"Total Nodes: {data.get('total_nodes')}")
            print(f"Total Links: {data.get('total_links')}")
            print(f"Total Documents: {data.get('total_documents')}")
            print(f"Nodes by Type: {data.get('nodes_by_fact_type')}")
            print("\n✅ Stats retrieved successfully!")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False


async def test_list_entities():
    """Test 6: List extracted entities."""
    print("\n" + "=" * 60)
    print("👥 TEST 6: List Extracted Entities")
    print("=" * 60)
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        response = await client.get(
            f"/v1/default/banks/{BANK_ID}/entities",
            params={"limit": 20}
        )
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            print(f"Found {len(items)} entities:")
            
            for entity in items[:10]:  # Show first 10
                name = entity.get("canonical_name", "Unknown")
                count = entity.get("mention_count", 0)
                print(f"  • {name} (mentioned {count} times)")
            
            print("\n✅ Entities retrieved successfully!")
            return True
        else:
            print(f"❌ Entities failed: {response.status_code}")
            return False


async def test_cleanup():
    """Test 7: Optional - Clean up test bank."""
    print("\n" + "=" * 60)
    print("🧹 TEST 7: Cleanup (Optional)")
    print("=" * 60)
    
    print("Skipping cleanup to preserve test data.")
    print("To delete test bank, uncomment the code below.")
    
    # Uncomment to actually delete:
    # async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
    #     response = await client.delete(f"/v1/default/banks/{BANK_ID}")
    #     print(f"Deleted: {response.json()}")
    
    return True


async def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "🚀" * 20)
    print("     MEMORY API TEST SUITE FOR SWE-AGENT")
    print("🚀" * 20)
    print(f"\nServer: {BASE_URL}")
    print(f"Test Bank: {BANK_ID}")
    print(f"Time: {datetime.now().isoformat()}")
    
    tests = [
        ("Health Check", test_health),
        ("Retain Code Patterns", test_retain_code_patterns),
        ("Recall Error Fixes", test_recall_error_fixes),
        ("Reflect on Problem", test_reflect_on_problem),
        ("Bank Statistics", test_bank_stats),
        ("List Entities", test_list_entities),
        ("View Embeddings (DB)", test_view_embeddings_via_db),
        ("Cleanup", test_cleanup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED WITH ERROR: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Memory API is ready for SWE-Agent integration.")
    else:
        print("\n⚠️ Some tests failed. Please check the server logs.")


async def run_single_test(test_name: str):
    """Run a single test by name."""
    test_map = {
        "health": test_health,
        "retain": test_retain_code_patterns,
        "recall": test_recall_error_fixes,
        "reflect": test_reflect_on_problem,
        "stats": test_bank_stats,
        "entities": test_list_entities,
        "embeddings": test_view_embeddings_via_db,
        "cleanup": test_cleanup,
    }
    
    if test_name in test_map:
        await test_map[test_name]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available: {list(test_map.keys())}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run single test: python test_api.py embeddings
        test_name = sys.argv[1]
        asyncio.run(run_single_test(test_name))
    else:
        # Run all tests
        asyncio.run(run_all_tests())