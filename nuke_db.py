import asyncio
from vectorstore.pgvector_client import get_pgvector_client

async def nuke():
    client = get_pgvector_client()
    await client.connect()
    try:
        async with client._pool.acquire() as conn:
            await conn.execute('TRUNCATE TABLE document_chunks CASCADE;')
            await conn.execute('TRUNCATE TABLE documents CASCADE;')
            print('✅ Database wiped clean!')
    except Exception as e: print(f'❌ Error: {e}')
    finally: await client.close()

if __name__ == '__main__': asyncio.run(nuke())