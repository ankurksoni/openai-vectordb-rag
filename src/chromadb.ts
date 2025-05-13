/**
 * This script demonstrates Retrieval-Augmented Generation (RAG) using ChromaDB and OpenAI.
 *
 * Flow:
 * 1. Embeds example data and stores it in ChromaDB.
 * 2. Prompts the user for a question.
 * 3. Finds the most relevant info using vector search.
 * 4. Uses OpenAI GPT to answer the question using the found context.
 */

import { ChromaClient, OpenAIEmbeddingFunction } from "chromadb";
import OpenAI from "openai";
const chroma = new ChromaClient({ path: "http://localhost:8000" });
import readline from 'readline';

// Example data to be embedded and stored
const studentInfo = `Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.`;

const clubInfo = `The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.`;

const universityInfo = `The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.`;

// Embedding function using OpenAI
const embeddingFunction: OpenAIEmbeddingFunction = new OpenAIEmbeddingFunction({
    openai_api_key: process.env.OPENAI_API_KEY!,
    openai_model: 'text-embedding-3-small'
})

const collectionName = "personal-infos";

/**
 * Creates a new collection in ChromaDB if it doesn't exist.
 */
async function createCollection() {
    await chroma.createCollection({ name: collectionName });
}

/**
 * Populates the ChromaDB collection with example documents and their embeddings.
 */
async function populateCollection() {
    const collection = await getCollection();
    await collection.add({
        documents: [studentInfo, clubInfo, universityInfo],
        ids: ['id1', 'id2', 'id3'],
    })
}

/**
 * Retrieves the ChromaDB collection, using the OpenAI embedding function.
 */
async function getCollection() {
    const collection = await chroma.getCollection({
        name: collectionName,
        embeddingFunction
    });
    return collection;
}

/**
 * Prompts the user for a question, retrieves the most relevant info from ChromaDB,
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

    const collection = await getCollection();
    const result = await collection.query({
        queryTexts: question,
        nResults: 1
    });
    const relevantInfo = result.documents[0][0];

    if (relevantInfo) {
        const openai = new OpenAI();
        const response = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            temperature: 0,
            messages: [{
                role: 'assistant',
                content: `Answer the next question using this information: ${relevantInfo}`
            },
            {
                role: 'user',
                content: question
            }]
        });
        const responseMessage = response.choices[0].message;
        console.log(responseMessage.content);
    } else {
        console.log("No relevant information found.");
    }
}

/**
 * Prepares the dataset by creating the collection and populating it.
 */
async function prepareDataSet() {
    await createCollection();
    await populateCollection();
}

/**
 * Main entry point: prepares the dataset and starts the Q&A loop.
 */
async function ask() {
    await askQuestion();
}

// Run the steps
prepareDataSet();
ask();