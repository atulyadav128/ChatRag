// Switch to your chatbot database
use('chat_rag_db');

// Create chat_history collection and insert a sample message
db.getCollection('chat_history').insertOne({
  session_id: "session_001",
  role: "user",
  message: "Hello, how are you?",
  timestamp: new Date()
});

// Create vectors collection and insert a sample vector
db.getCollection('vectors').insertOne({
  doc_id: "doc_001",
  text: "Sample document text for embedding.",
  vector: [0.12, 0.34, 0.56, 0.78], // Example vector
  metadata: { source: "manual" }
});