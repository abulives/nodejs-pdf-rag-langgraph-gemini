const fs = require('fs');
require('dotenv').config();
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { StateGraph } = require("@langchain/langgraph");
const { z } = require("zod");
const { tool } = require("@langchain/core/tools");
const retrieveSchema = z.object({ query: z.string() });
const { AIMessage, HumanMessage, SystemMessage, ToolMessage } = require("@langchain/core/messages");
const { MessagesAnnotation } = require("@langchain/langgraph");
const { ToolNode } = require("@langchain/langgraph/prebuilt");
const { toolsCondition } = require("@langchain/langgraph/prebuilt");

const apiKey = process.env.GOOGLE_API_KEY;
const embeddings = new GoogleGenerativeAIEmbeddings({ model: "models/embedding-001", apiKey });

const model = new ChatGoogleGenerativeAI({
    model: 'gemini-2.0-flash',
    apiKey,
    temperature: 0.3,
});

const getPdfText = async (files) => {
    try {
        const docs = [];
        for (const file of files) {
            const loader = new PDFLoader(file.path);
            const fileDocs = await loader.load();
            docs.push(...fileDocs);
        }
        return docs;
    } catch (error) {
        throw error;
    }
};

const getTextChunks = async (docs) => {
    try {
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splits = await splitter.splitDocuments(docs);
        return splits;
    } catch (error) {
        throw error;
    }
};

const getVectorStore = async (text_chunks) => {
    try {
        const vectorStore = await FaissStore.fromDocuments(text_chunks, embeddings);
        const vectorname = 'uploaded_vectors';
        await vectorStore.save(`vectors/${vectorname}`);
        return;
    } catch (error) {
        console.error("Error during fromTexts:", error.message);
        throw error;
    }
};
const getConversationalChain = async () => {
    return async ({ question }) => {
        try {
            const retrieve = tool(
                async ({ query }) => {
                    const vectorname = 'uploaded_vectors';
                    const vectorStore = await FaissStore.load(`vectors/${vectorname}`, embeddings);
                    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
                    const serialized = retrievedDocs
                        .map(doc => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`)
                        .join("\n");
                    return [serialized, retrievedDocs];
                },
                {
                    name: "retrieve",
                    description: "Retrieve information related to a query.",
                    schema: retrieveSchema,
                    responseFormat: "content_and_artifact",
                }
            );

            // Node: Determine whether to call tool or respond directly
            async function queryOrRespond(state) {
                const llmWithTools = model.bindTools([retrieve]);
                const response = await llmWithTools.invoke(state.messages);
                return { messages: [response] };
            }

            // Node: Use retrieved results to generate the final answer
            async function generate(state) {
                const recentToolMessages = state.messages.filter(msg => msg instanceof ToolMessage);
                const docsContent = recentToolMessages.map(doc => doc.content).join("\n");

                const systemMessageContent =
                    "You are an assistant for question-answering tasks. " +
                    "Use the following pieces of retrieved context to answer " +
                    "the question. If you don't know the answer, say that you " +
                    "don't know. Use three sentences maximum and keep the " +
                    "answer concise.\n\n" +
                    docsContent;

                const conversationMessages = state.messages.filter(
                    msg =>
                        msg instanceof HumanMessage ||
                        msg instanceof SystemMessage ||
                        (msg instanceof AIMessage && msg.tool_calls?.length === 0)
                );

                const prompt = [
                    new SystemMessage({ content: systemMessageContent }),
                    ...conversationMessages,
                ];

                const response = await model.invoke(prompt);
                return { messages: [response] };
            }

            const tools = new ToolNode([retrieve]);

            const graphBuilder = new StateGraph(MessagesAnnotation)
                .addNode("queryOrRespond", queryOrRespond)
                .addNode("tools", tools)
                .addNode("generate", generate)
                .addEdge("__start__", "queryOrRespond")
                .addConditionalEdges("queryOrRespond", toolsCondition, {
                    __end__: "__end__",
                    tools: "tools",
                })
                .addEdge("tools", "generate")
                .addEdge("generate", "__end__");

            const graph = graphBuilder.compile();

            const result = await graph.invoke({
                messages: [new HumanMessage({ content: question })],
            });

            const lastMessage = result.messages?.[result.messages.length - 1];

            return { output_text: lastMessage?.content || '' };
        } catch (error) {
            console.error("Error calling the model:", error.message);
            throw error;
        }
    };
};

const userInput = async (userQuestion) => {
    try {
        const chain = await getConversationalChain();
        const response = await chain({ question: userQuestion });
        return response;
    } catch (err) {
        throw new Error(err);
    }
};

module.exports = {
    uploadPDFs: async (files) => {
        try {
            const rawText = await getPdfText(files);
            const textChunks = await getTextChunks(rawText);
            await getVectorStore(textChunks);
            return true;
        } catch (err) {
            throw new Error(err);
        }
    },
    askQuestions: async (question) => {
        try {
            const response = await userInput(question);
            return { success: true, response: response.output_text };
        } catch (err) {
            console.error('Error in getQuestions:', err);
            return { success: false, error: err.message || 'An error occurred' };
        }
    }
};
