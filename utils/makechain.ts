import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Étant donné la conversation suivante et une question de suivi, si vous ne pouvez pas répondre avec succès à la question en vous servant de l'historique, reformulez la question de suivi pour qu'elle soit une question indépendante. Répondez en français.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Vous êtes un assistant IA fournissant des conseils utiles. On vous donne les extraits suivants d'une base de connaissance et une question. Fournissez une réponse conversationnelle basée sur le contexte fourni. Si ça ne suffit pas, basez vous sur l'historique de la conversation.
Si vous ne pouvez pas trouver la réponse dans le contexte ci-dessous, dites simplement "Je ne suis pas sûr d'avoir compris votre requête, pouvez-vous la reformuler ?". N'essayez pas de donner une réponse inventée.
Si la question n'est pas liée au contexte de la base de connaissances, répondez poliment que vous êtes programmé pour répondre uniquement aux questions qui sont liées au contexte.

Question: {question}
=========
{context}
=========
Réponse en langage md avec une mise en forme agréable:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.3,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: false,
    k: 5, //number of source documents to return
  });
};
