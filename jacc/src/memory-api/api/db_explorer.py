"""
Database Explorer - View all schemas, tables, banks, and data in Memory API

Run from: memory-api/api directory
Command: python db_explorer.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

DATABASE_URL = os.getenv("MEMORY_API_DATABASE_URL", "postgresql://memory:memory_secret@localhost:5432/memory_db")


async def explore_database():
    import asyncpg
    
    print("\n" + "=" * 70)
    print("🔍 DATABASE EXPLORER - Memory API")
    print("=" * 70)
    print(f"URL: {DATABASE_URL.replace(DATABASE_URL.split(':')[2].split('@')[0], '***')}")
    
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # 1. List all schemas
        print("\n" + "-" * 70)
        print("📁 SCHEMAS")
        print("-" * 70)
        schemas = await conn.fetch("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            ORDER BY schema_name
        """)
        for s in schemas:
            print(f"  • {s['schema_name']}")
        
        # 2. List all tables per schema
        print("\n" + "-" * 70)
        print("📋 TABLES BY SCHEMA")
        print("-" * 70)
        tables = await conn.fetch("""
            SELECT table_schema, table_name, 
                   pg_size_pretty(pg_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name))) as size
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """)
        
        current_schema = None
        for t in tables:
            if t['table_schema'] != current_schema:
                current_schema = t['table_schema']
                print(f"\n  [{current_schema}]")
            print(f"    • {t['table_name']} ({t['size']})")
        
        # 3. List banks (tenants)
        print("\n" + "-" * 70)
        print("🏦 BANKS (Memory Tenants)")
        print("-" * 70)
        try:
            banks = await conn.fetch("""
                SELECT id, created_at, updated_at, 
                       COALESCE(json_extract_path_text(disposition::json, 'name'), 'N/A') as name
                FROM public.banks
                ORDER BY created_at DESC
                LIMIT 20
            """)
            if banks:
                for b in banks:
                    print(f"  • {b['id']}")
                    print(f"    Created: {b['created_at']}")
            else:
                print("  (No banks found)")
        except Exception as e:
            print(f"  ⚠️ Could not list banks: {e}")
        
        # 4. Memory units summary
        print("\n" + "-" * 70)
        print("🧠 MEMORY UNITS SUMMARY")
        print("-" * 70)
        try:
            summary = await conn.fetch("""
                SELECT bank_id, fact_type, COUNT(*) as count
                FROM public.memory_units
                GROUP BY bank_id, fact_type
                ORDER BY bank_id, count DESC
            """)
            if summary:
                current_bank = None
                for s in summary:
                    if s['bank_id'] != current_bank:
                        current_bank = s['bank_id']
                        print(f"\n  Bank: {current_bank}")
                    print(f"    • {s['fact_type']}: {s['count']} units")
            else:
                print("  (No memory units found)")
        except Exception as e:
            print(f"  ⚠️ Could not list memory units: {e}")
        
        # 5. Documents summary
        print("\n" + "-" * 70)
        print("📄 DOCUMENTS")
        print("-" * 70)
        try:
            docs = await conn.fetch("""
                SELECT bank_id, COUNT(*) as count, 
                       SUM(COALESCE(chunk_count, 0)) as total_chunks
                FROM public.documents
                GROUP BY bank_id
                ORDER BY count DESC
            """)
            if docs:
                for d in docs:
                    print(f"  • Bank {d['bank_id']}: {d['count']} docs, {d['total_chunks'] or 0} chunks")
            else:
                print("  (No documents found)")
        except Exception as e:
            print(f"  ⚠️ Could not list documents: {e}")
        
        # 6. Entities summary
        print("\n" + "-" * 70)
        print("👥 ENTITIES (Top 10)")
        print("-" * 70)
        try:
            entities = await conn.fetch("""
                SELECT canonical_name, mention_count, bank_id
                FROM public.entities
                ORDER BY mention_count DESC
                LIMIT 10
            """)
            if entities:
                for e in entities:
                    print(f"  • {e['canonical_name']} ({e['mention_count']} mentions) - Bank: {e['bank_id']}")
            else:
                print("  (No entities found)")
        except Exception as e:
            print(f"  ⚠️ Could not list entities: {e}")
        
        # 7. Sample embeddings
        print("\n" + "-" * 70)
        print("🔢 SAMPLE EMBEDDINGS (Last 5)")
        print("-" * 70)
        try:
            embeddings = await conn.fetch("""
                SELECT 
                    id,
                    bank_id,
                    fact_type,
                    LEFT(text, 60) as text_preview,
                    array_length(embedding::real[], 1) as dim,
                    created_at
                FROM public.memory_units
                ORDER BY created_at DESC
                LIMIT 5
            """)
            if embeddings:
                for e in embeddings:
                    print(f"\n  ID: {e['id'][:8]}...")
                    print(f"    Bank: {e['bank_id']}")
                    print(f"    Type: {e['fact_type']}")
                    print(f"    Text: {e['text_preview']}...")
                    print(f"    Embedding: {e['dim']} dimensions")
            else:
                print("  (No embeddings found)")
        except Exception as e:
            print(f"  ⚠️ Could not list embeddings: {e}")
        
        # 8. Database size
        print("\n" + "-" * 70)
        print("💾 DATABASE SIZE")
        print("-" * 70)
        size = await conn.fetchrow("""
            SELECT pg_size_pretty(pg_database_size(current_database())) as size
        """)
        print(f"  Total: {size['size']}")
        
    finally:
        await conn.close()
    
    print("\n" + "=" * 70)
    print("✅ Database exploration complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(explore_database())
