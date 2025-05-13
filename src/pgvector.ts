/**
 * This script demonstrates Retrieval-Augmented Generation (RAG) using Postgres with the pgvector extension and OpenAI.
 *
 * Flow:
 * 1. Embeds example data and stores it in a Postgres table with vector support.
 * 2. Prompts the user for a question.
 * 3. Finds the most relevant info using vector similarity search (pgvector).
 * 4. Uses OpenAI GPT to answer the question using the found context.
 */

import { Client } from "pg";
import pgvector from "pgvector/pg";

import OpenAI from "openai";
import readline from 'readline';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pg = new Client({
  connectionString: process.env.DATABASE_URL, // use your DB string
});
await pg.connect();

await pgvector.registerTypes(pg);

// Example data to be embedded and stored
const studentInfo = `Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA...`;
const clubInfo = `The university chess club provides an outlet for students to come together...`;
const universityInfo = `The University of Washington, founded in 1861 in Seattle, is a public research university...`;

/**
 * Ensures the pgvector extension is enabled and the documents table is created.
 * Drops the table if it exists for a fresh start.
 */
async function setupTable() {
  await pg.query(`CREATE EXTENSION IF NOT EXISTS vector;`);
  await pg.query('DROP TABLE IF EXISTS documents');
  await pg.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id serial PRIMARY KEY,
      content TEXT,
      embedding vector(1536)
    );
  `);
}

/**
 * Embeds the given content using OpenAI and inserts it into the documents table.
 */
async function embedAndInsert(content: string) {
  const embedding = await getEmbedding(content);
  const query = `INSERT INTO documents (content, embedding) VALUES ($1, $2);`;
  await pg.query(query, [content, pgvector.toSql(embedding)]);
}

/**
 * Gets the embedding vector for a given text using OpenAI.
 */
async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  return res.data[0].embedding;
}

/**
 * Finds the most relevant document in the table for the given question using vector similarity.
 */
async function queryRelevantDocument(question: string): Promise<string | null> {
  const embedding = await getEmbedding(question);
  const { rows } = await pg.query(
    `SELECT content FROM documents
     ORDER BY embedding <-> $1
     LIMIT 1;`,
    [pgvector.toSql(embedding)]
  );
  return rows[0]?.content || null;
}

/**
 * Prompts the user for a question, retrieves the most relevant info from Postgres,
 * and uses OpenAI GPT to answer the question using that context.
 */
async function askQuestion() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const question = await new Promise<string>((resolve) => {
    rl.question('What would you like to ask? ', (answer: string) => {
      rl.close();
      resolve(answer);
    });
  });

  const context = await queryRelevantDocument(question);
  if (!context) return console.log("No relevant document found.");

  const res = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    temperature: 0,
    messages: [
      { role: "assistant", content: `Use this information: ${context}` },
      { role: "user", content: question },
    ],
  });

  console.log(res.choices[0].message.content);
}

// ----- RUN THE STEPS -----
// 1. Set up the table
// 2. Insert example data
// 3. Prompt user for a question and answer it using RAG
await setupTable();
await embedAndInsert(studentInfo);
await embedAndInsert(clubInfo);
await embedAndInsert(universityInfo);
await askQuestion();

await pg.end();
