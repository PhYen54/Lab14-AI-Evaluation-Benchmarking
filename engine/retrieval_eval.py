import logging
from typing import List, Dict

# Set up logging for detailed case reports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Evaluates the performance of the Retrieval stage in a RAG pipeline.
    Calculates metrics like Hit Rate and Mean Reciprocal Rank (MRR).
    """
    def __init__(self, vector_db=None):
        """
        Initialize with an optional vector_db connection.
        """
        self.vector_db = vector_db

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Check if at least one of the expected_ids is within the top_k retrieved_ids.
        Returns 1.0 if any expected_id is found, 0.0 otherwise.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
        
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        MRR = 1 / position of the first relevant document (1-indexed).
        If no relevant document is found, returns 0.0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Calculates retrieval metrics for a single test case. 
        Matches engine/runner.py 'self.evaluator.score(test_case, response)'.
        
        Expected fields:
        - test_case: contains 'expected_retrieval_ids'
        - response: contains 'retrieved_ids' or in 'metadata'
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        
        # Priority: response['retrieved_ids'] > response['metadata']['retrieved_ids']
        retrieved_ids = response.get("retrieved_ids", [])
        if not retrieved_ids and "metadata" in response:
            retrieved_ids = response["metadata"].get("retrieved_ids", [])

        # If retrieved_ids is still empty, maybe it's in 'contexts' (fallback simulation)
        if not retrieved_ids and "contexts" in response:
            # Simulation: map contexts to dummy IDs if needed, but per plan, IDs should be there
            pass

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        # Detailed logging for failure analysis (Requirement M2)
        logger.info(f"Evaluating Case: {test_case.get('question', 'N/A')[:50]}...")
        logger.info(f"  - Expected IDs: {expected_ids}")
        logger.info(f"  - Retrieved IDs: {retrieved_ids[:5]} (Top 5)")
        logger.info(f"  - Hit Rate@3: {hit_rate}, MRR: {mrr}")
        
        # Report correct vs incorrect retrievals
        correct_retrievals = [rid for rid in retrieved_ids if rid in expected_ids]
        if correct_retrievals:
            logger.info(f"  - Correct IDs found: {correct_retrievals}")
        else:
            logger.warning(f"  - NO expected IDs found in retrieval result!")

        return {
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids
            },
            # Placeholders for RAGAS compatibility in runner.py
            "faithfulness": 1.0 if hit_rate > 0 else 0.0, 
            "relevancy": hit_rate
        }

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Evaluates retrieval performance across the entire dataset.
        Returns aggregate metrics.
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        num_cases = len(dataset)

        if num_cases == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        for i, case in enumerate(dataset):
            # In evaluate_batch, we assume the dataset item has both expected and actual
            expected = case.get("expected_retrieval_ids", [])
            retrieved = case.get("retrieved_ids", [])
            
            hr = self.calculate_hit_rate(expected, retrieved)
            mrr = self.calculate_mrr(expected, retrieved)
            
            total_hit_rate += hr
            total_mrr += mrr
            
            logger.debug(f"Case {i+1}: HR={hr}, MRR={mrr}")

        avg_hit_rate = total_hit_rate / num_cases
        avg_mrr = total_mrr / num_cases

        logger.info(f"Batch Evaluation Results:")
        logger.info(f"  - Total Cases: {num_cases}")
        logger.info(f"  - Avg Hit Rate: {avg_hit_rate:.4f}")
        logger.info(f"  - Avg MRR: {avg_mrr:.4f}")

        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "total_cases": num_cases
        }
