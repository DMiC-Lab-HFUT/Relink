import redis
from utils.neo4j_operator import *
from utils.my_llm import LLMClient, ClientConfig
from configs import *
from utils.vector_store import MyVectorStore
from core.llm_functions import eval_sufficiency_with_llm, missing_knowledge_identify, missing_knowledge_extraction
from configs import logger
from utils.embeddings import CustomEmbedding

class KnowledgeFeedback:
    def __init__(self):
        """Graph database retriever initialization"""
        self.init_config()
        self.feedback_knowledge = {
            'entities': [],
            'relations': []
        }

    def init_config(self):
        #  redis
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(endpoints=LLM_CONFIG['endpoints'], client_cfg=ClientConfig(cache=self.redis_client))

        self.feedback_base = Neo4jOperations(
            uri=FEEDBACK_BASE["uri"],
            auth=FEEDBACK_BASE["auth"]
        )


        self.embedding_model = CustomEmbedding(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url'],
            model_name=EMBEDDING_CONFIG['model_name']
        )

        self.doc_vector_db = MyVectorStore(
            DATASET_CONFIG['doc_vector_store_path'],
            embedding=self.embedding_model
        )




    def _format_context(self, paths):
        # Extract triples and format them,while maintaining order and removing duplicates
        triples = [
            f"<{triple['begin']['mention']} | {triple['r'].replace('_', ' ')} | {triple['end']['mention']}>"
            for path in paths for triple in path.get('relations', [])
        ]
        format_triples = list(dict.fromkeys(triples))

        # Extract relevant sentences,maintaining order and removing duplicates.
        related_sents = [
            sent for path in paths for sent in path.get('context_sentences', [])
        ]
        related_sents = list(dict.fromkeys(related_sents))

        # Concatenate into a string,adding the title only when there is content.
        parts = []
        if format_triples:
            parts.append("Triples:\n" + "\n".join(format_triples))
        if related_sents:
            parts.append("Related Text:\n" + "\n".join(related_sents))
        context_info = "\n\n".join(parts)
        return context_info

    def refine_kg(self, domain, entities, relations):
        eid2id = {}
        logger.info(f'Refined KG with new entities: {len(entities)}')
        for ent in entities:
            e_id = hash_text(ent['mention'].lower().strip())
            eid2id[ent['id']] = e_id
            ent['id'] = e_id
            self.feedback_base.create_or_merge_node(domain, ent)

        logger.info(f'Refined KG with new relations: {len(relations)}')
        for rel in relations:
            self.feedback_base.create_or_update_relationship(
                head_id= eid2id[rel['source']],
                tail_id= eid2id[rel['target']],
                rel_type=rel['relation'],
                properties={"evidence": rel['evidence']},
                head_label=domain,
                tail_label=domain,
            )

    def feedback_to_knowledge_base(self, domain, answer, paths, question, step):
        if not eval_sufficiency_with_llm(self.llm, paths, question):


            # 1) Identify the missing information and present it in the form of multiple sub-questions.
            sub_questions = missing_knowledge_identify(self.llm, question, paths)

            # 2) Extract the corresponding triples from the original document based on the missing information.
            contexts = []
            for q in sub_questions:
                contexts.extend([d.page_content for d in self.doc_vector_db.query_collection(domain, q, 8)])

            context = "\n".join(list(dict.fromkeys(contexts)))
            entities, relations = missing_knowledge_extraction(self.llm, question, paths, "\n".join(sub_questions), context)
            self.refine_kg(domain, entities, relations)
            pass
        return

