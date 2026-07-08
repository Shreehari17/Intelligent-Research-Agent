from core.db import get_connection, release_connection

conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, tenant_id, summary FROM memories")
rows = cursor.fetchall()
for row in rows:
    print(row)
cursor.close()
release_connection(conn)