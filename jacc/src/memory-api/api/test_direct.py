"""
Direct Memory Engine Test (No HTTP Server Required)

This script tests the MemoryEngine directly by importing it,
bypassing the HTTP layer entirely. Perfect for development!

Run from: memory-api/api directory
Command: python test_direct.py
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("../.env")

# MEMORY_API_DATABASE_URL should be set in .env file
# If not, this will fail with a clear error


async def main():
    print("\n" + "=" * 60)
    print("🔧 DIRECT MEMORY ENGINE TEST (No HTTP)")
    print("=" * 60)
    
    # Import after setting env vars
    from src import MemoryEngine
    from src.models import RequestContext
    from src.engine.memory_engine import Budget
    
    # Create engine
    print("\n[1] Creating MemoryEngine...")
    engine = MemoryEngine()
    
    print("[2] Initializing (connecting to database, running migrations)...")
    await engine.initialize()
    print("    ✅ Initialized!")
    
    # Health check
    print("\n[3] Health check...")
    health = await engine.health_check()
    print(f"    Status: {health}")
    
    # Retain a memory
    print("\n[4] Storing a code pattern...")
    bank_id = "dev_test"
    # ctx = RequestContext()
    
    # await engine.retain_batch_async(
    #     bank_id=bank_id,
    #     contents=[
    #     {
    #         "content": "When encountering ImportError in Python, first check if the module is installed with 'pip list | grep module_name'. If not installed, use 'pip install module_name'. If installed but still errors, check PYTHONPATH or virtual environment activation.",
    #         "context": "error_fix",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "TypeError: 'NoneType' object is not subscriptable - This error occurs when trying to access an index or key on a None value. Fix by adding null checks: 'if result is not None: result[key]' or using 'result.get(key) if result else default'.",
    #         "context": "error_fix",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "In Django projects, when tests fail with database errors, ensure test database migrations are applied. Run 'python manage.py migrate --run-syncdb' before running tests. Also check if the test is using the correct database settings.",
    #         "context": "testing",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "For Flask applications, when debugging routing issues, use 'app.url_map' to see all registered routes. Also enable debug mode with 'app.run(debug=True)' to get detailed error pages.",
    #         "context": "debugging",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "When fixing bugs in Python async code, remember that 'await' can only be used inside async functions. If getting 'SyntaxError: await outside async function', wrap the code in an async function and call it with asyncio.run().",
    #         "context": "async_python",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "Git conflict resolution pattern: First run 'git status' to see conflicted files. Open each file and look for <<<<<<< HEAD markers. Keep the correct version, remove conflict markers, then 'git add' and 'git commit'.",
    #         "context": "version_control",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "When a pytest test passes locally but fails in CI, check for: 1) Different Python versions 2) Missing environment variables 3) File path differences (Windows vs Linux) 4) Database state issues 5) Timing-dependent tests.",
    #         "context": "ci_debugging",
    #         "timestamp": datetime.now().isoformat()
    #     },
    #     {
    #         "content": "For pandas DataFrame operations with KeyError, first check column names with 'df.columns.tolist()'. Common issues: trailing spaces in column names, case sensitivity, or column was renamed/dropped earlier in the code.",
    #         "context": "data_processing",
    #         "timestamp": datetime.now().isoformat()
    #     }
    # ],
    #     request_context=ctx
    # )
    # print("    ✅ Stored!")
    
    # Recall
    # print("\n[5] Recalling memories...")
    # result = await engine.recall_async(
    #     bank_id=bank_id,
    #     query="How to debug async Python?",
    #     budget=Budget.LOW,
    #     fact_type=["world", "experience"],
    #     request_context=ctx
    # )
    
    # print(f"    Found {len(result.results)} results:")
    # for i, fact in enumerate(result.results[:3], 1):
    #     print(f"\n    {i}. {fact.text[:100]}...")
    #     print(f"       Type: {fact.fact_type}, Context: {fact.context}")
    
    # View embedding (direct database access)
    print("\n[6] Viewing embeddings directly...")
    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        # First, find out which schema the data is in
        schema_rows = await conn.fetch("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_name = 'memory_units'
        """)
        
        if schema_rows:
            schema_name = schema_rows[0]['table_schema']
            print(f"    Found memory_units in schema: {schema_name}")
            
            # pgvector needs cast to real[] before subscripting
            rows = await conn.fetch(f"""
                SELECT 
                    text,
                    (embedding::real[])[1:5] as preview,
                    array_length(embedding::real[], 1) as dim
                FROM {schema_name}.memory_units
                WHERE bank_id = $1
                LIMIT 3
            """, bank_id)
            
            for row in rows:
                print(f"\n    Text: {row['text'][:60]}...")
                print(f"    Embedding dim: {row['dim']}")
                if row['preview']:
                    print(f"    Preview: {[round(x, 4) for x in row['preview']]}")
        else:
            print("    ⚠️ memory_units table not found - skipping embedding view")
    
    # Cleanup
    print("\n[7] Closing engine...")
    await engine.close()
    print("    ✅ Done!")
    
    print("\n" + "=" * 60)
    print("✅ ALL DIRECT TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
