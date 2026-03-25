# fix_db.py
import asyncio
from vectorstore.pgvector_client import get_pgvector_client

async def fix_constraints():
    client = get_pgvector_client()
    await client.connect()
    try:
        async with client._pool.acquire() as conn:
            # Drop the global constraint
            await conn.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_hash_key;")
            # Add the per-user constraint (ignores errors if it already exists)
            await conn.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_user_hash_key') THEN 
                        ALTER TABLE documents ADD CONSTRAINT documents_user_hash_key UNIQUE (user_id, doc_hash); 
                    END IF; 
                END $$;
            """)
            print("✅ Database constraints permanently fixed!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(fix_constraints())