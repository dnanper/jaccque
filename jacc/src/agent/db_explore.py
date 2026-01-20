import asyncio
import asyncpg

async def view_all():
    conn = await asyncpg.connect('postgresql://memory:memory_secret@localhost:5432/memory_db')
    
    print('=' * 70)
    print('ALL MEMORY UNITS')
    print('=' * 70)
    
    # Count by bank
    counts = await conn.fetch('''
        SELECT bank_id, fact_type, COUNT(*) as count
        FROM public.memory_units
        GROUP BY bank_id, fact_type
        ORDER BY bank_id, count DESC
    ''')
    
    print('\n📊 Summary:')
    current_bank = None
    for c in counts:
        if c["bank_id"] != current_bank:
            current_bank = c["bank_id"]
            print(f'\n  Bank: {current_bank}')
        print(f'    • {c["fact_type"]}: {c["count"]}')
    
    # All units
    print('\n' + '=' * 70)
    print('📝 All Memory Units (newest first):')
    print('=' * 70)
    
    rows = await conn.fetch('''
        SELECT bank_id, fact_type, text, context, created_at
        FROM public.memory_units
        ORDER BY created_at DESC
        LIMIT 50
    ''')
    
    for i, r in enumerate(rows, 1):
        bank = r["bank_id"]
        fact_type = r["fact_type"]
        text = r["text"][:300] if r["text"] else ""
        context = r["context"]
        created = r["created_at"]
        
        print(f'\n[{i}] [{bank}] {fact_type.upper()}')
        print(f'    Context: {context}')
        print(f'    Created: {created}')
        print(f'    Text: {text}...')
        print('-' * 50)
    
    await conn.close()
    print('\n' + '=' * 70)
    print(f'Total shown: {len(rows)} units')
    print('=' * 70)

asyncio.run(view_all())
