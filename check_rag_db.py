import sqlite3

conn = sqlite3.connect("rag_docs.db")
rows = conn.execute("SELECT id, content FROM documents").fetchall()
print("Documents in DB:")
for row in rows:
    print(row)
conn.close()
