#!/usr/bin/env python3
"""
Script to test GPT model on Q&A pairs from the World History to 1500 dataset.
Loads 20 random Q&A pairs, tests GPT-5 (or specified model) on each question,
evaluates correctness, and prints accuracy.
"""

import json
import random
import requests
from openai import OpenAI
import os
from typing import List, Dict, Tuple

# Configuration
DATASET_URL = "https://raw.githubusercontent.com/provos/world-history-to-1500-qa/master/dataset.json"
MODEL_NAME = "gpt-4o"  # Using gpt-4o as GPT-5 is not yet available. Change to "gpt-5" when available.
NUM_PAIRS = 5


def load_dataset(url: str) -> Dict:
    """Load the dataset from the GitHub URL."""
    print(f"Loading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_random_qa_pairs(dataset: Dict, num_pairs: int) -> List[Dict]:
    """Select random Q&A pairs from the dataset."""
    qa_pairs = dataset.get("qa_pairs", [])
    if len(qa_pairs) < num_pairs:
        print(f"Warning: Dataset only has {len(qa_pairs)} pairs, using all available.")
        return qa_pairs
    return random.sample(qa_pairs, num_pairs)


def get_gpt_answer(client: OpenAI, model: str, question: str) -> str:
    """Get an answer from GPT for the given question."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about world history. Answer completely in about 100 words."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting GPT answer: {e}")
        return ""


def extract_facts(client: OpenAI, model: str, question: str, answer: str) -> List[str]:
    """Extract facts from an answer using an LLM."""
    extraction_prompt = f"""You are a fact extraction expert. Extract all factual claims which help answer the question from the following answer.
Do not evaluate the facts, just extract them if you think they were intended to answer the question.

Question: {question}
Answer: {answer}

Extract each distinct factual claim as a separate fact. Each fact should be a clear, atomic statement that conveys one piece of information.

Return the facts as a JSON object with a "facts" key containing an array of strings, where each string is one fact. Example format:
{{"facts": ["fact 1", "fact 2", "fact 3"]}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fact extraction expert. Return facts as a JSON object with a 'facts' array."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        # Try to parse JSON response
        try:
            result = json.loads(content)
            # Handle both {"facts": [...]} and direct array formats
            if isinstance(result, dict) and "facts" in result:
                facts = result["facts"]
                return facts if isinstance(facts, list) else []
            elif isinstance(result, list):
                return result
            else:
                # If it's a dict but not with "facts" key, try to find array values
                for key, value in result.items():
                    if isinstance(value, list):
                        return value
                return []
        except json.JSONDecodeError:
            # Fallback: try to extract facts from text format
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            facts = []
            for line in lines:
                if line.startswith('"') or line.startswith('-') or line.startswith('*'):
                    fact = line.strip('"').strip('-').strip('*').strip()
                    if fact:
                        facts.append(fact)
            return facts if facts else [content]
    except Exception as e:
        print(f"Error extracting facts from answer to question: {question}. Error: {e}")
        return []


def check_contradictions(client: OpenAI, model: str, correct_facts: List[str], gpt_facts: List[str]) -> List[str]:
    """Check for contradictions between correct answer facts and GPT answer facts."""
    if not correct_facts or not gpt_facts:
        return []
    
    contradiction_prompt = f"""You are a contradiction detection expert. Compare the facts from answer 1 with the facts from answer 2 and identify any contradictions.
Only report contradictions that clearly can not both be true.

Correct Answer Facts:
{json.dumps(correct_facts, indent=2)}

GPT Answer Facts:
{json.dumps(gpt_facts, indent=2)}

Identify pairs of facts that directly contradict each other. A contradiction means one fact explicitly contradicts or negates the other.

Return the contradictions as a JSON object with a "contradictions" key containing an array of strings, where each string describes one contradiction. If there are no contradictions, return an empty array.

Example format:
{{"contradictions": ["Fact X from correct answer contradicts Fact Y from GPT answer"]}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a contradiction detection expert. Return contradictions as a JSON object with a 'contradictions' array."},
                {"role": "user", "content": contradiction_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
            if isinstance(result, dict) and "contradictions" in result:
                return result["contradictions"]
            elif isinstance(result, list):
                return result
            else:
                return []
        except json.JSONDecodeError:
            return []
    except Exception as e:
        print(f"Error checking contradictions: {e}")
        return []


def check_fact_coverage(client: OpenAI, model: str, correct_facts: List[str], gpt_facts: List[str]) -> Tuple[bool, List[str]]:
    """Check if all facts from correct answer have synonyms in GPT answer."""
    if not correct_facts:
        return True, []
    
    coverage_prompt = f"""You are a semantic equivalence expert. Check if every fact from the answer 1 has a corresponding fact (synonym or equivalent) in the answer 2.

Answer 1 Facts:
{json.dumps(correct_facts, indent=2)}

Answer 2 Facts:
{json.dumps(gpt_facts, indent=2)}

For each fact in the answer 1, determine if there is a fact in the answer 2 that expresses the same information (even if worded differently or using synonyms).

Return a JSON object with:
- "facts_covered": array of facts from answer 1 that have equivalents in answer 2
- "missing_facts": array of facts from answer 1 that do NOT have equivalents in answer 2

Example format:
{{"facts_covered": ["fact that is covered"], "missing_facts": ["fact that is missing"]}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a semantic equivalence expert. Return results as a JSON object only."},
                {"role": "user", "content": coverage_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
            facts_covered = result.get("facts_covered", [])
            missing_facts = result.get("missing_facts", [])
            return facts_covered, missing_facts if isinstance(missing_facts, list) else []
        except json.JSONDecodeError:
            return [], correct_facts  # Conservative: assume not all covered if parsing fails
    except Exception as e:
        print(f"Error checking fact coverage: {e}")
        return [], correct_facts  # Conservative: assume not all covered on error


def evaluate_answer(client: OpenAI, model: str, question: str, correct_answer: str, gpt_answer: str) -> Tuple[bool, Dict]:
    """Evaluate whether GPT's answer is correct by extracting facts and comparing them.
    
    Returns:
        Tuple of (is_correct: bool, evaluation_details: dict)
    """
    # Extract facts from both answers
    print("    Extracting facts from correct answer...")
    correct_facts = extract_facts(client, model, question, correct_answer)
    
    print("    Extracting facts from GPT answer...")
    gpt_facts = extract_facts(client, model, question, gpt_answer)
    # Check for contradictions
    print("    Checking for contradictions...")
    contradictions = check_contradictions(client, model, correct_facts, gpt_facts)
    print(contradictions)
    
    # Check if all facts from correct answer are covered
    print("    Checking fact coverage...")
    facts_covered, missing_facts = check_fact_coverage(client, model, correct_facts, gpt_facts)
    print(f'len(facts_covered): {len(facts_covered)}')
    print(f'len(missing_facts): {len(missing_facts)}')
    print(missing_facts)
    
    # Determine correctness: no contradictions AND all facts covered
    score = 0 if len(contradictions) != 0 else len(facts_covered) / (len(facts_covered) + len(missing_facts))
    
    evaluation_details = {
        "correct_facts": correct_facts,
        "gpt_facts": gpt_facts,
        "contradictions": contradictions,
        "facts_covered": facts_covered,
        "missing_facts": missing_facts,
    }
    
    return score, evaluation_details
def evaluate_with_fact_checking(client: OpenAI, model: str, qa_pairs: List[Dict]):
    # Test each pair
    results = []
    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        correct_answer = pair["answer"]
        
        print(f"[{i}/{len(qa_pairs)}] Question: {question[:100]}...")
        
        # Get GPT answer
        print("  Getting GPT answer...")
        gpt_answer = get_gpt_answer(client, model, question)
        
        if not gpt_answer:
            print("  Warning: Failed to get GPT answer, skipping...")
            continue
        
        # Evaluate correctness
        print("  Evaluating answer...")
        print(f"  Correct answer: {correct_answer}")
        print(f"  GPT answer: {gpt_answer}")
        score, evaluation_details = evaluate_answer(client, model, question, correct_answer, gpt_answer)
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "gpt_answer": gpt_answer,
            "score": score,
            "evaluation_details": evaluation_details
        })
        
        status = str(score)
        print(f"  Result: {status}")
        if evaluation_details.get("contradictions"):
            print(f"    Contradictions found: {len(evaluation_details['contradictions'])}")
        if evaluation_details.get("missing_facts"):
            print(f"    Missing facts: {len(evaluation_details['missing_facts'])}")
        print()
    
    # Calculate and print accuracy
    if results:
        average_score = sum(r["score"] for r in results) / len(results)
        accuracy = average_score * 100
        
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total questions tested: {len(results)}")
        print(f"Average score: {average_score:.2f}")
        print(f'Number of contradicting answers: {sum([1 if len(result["evaluation_details"]["contradictions"] ) != 0 else 0 for result in results])}')
        print(f'Number of answers with missing facts: {sum([1 if len(result["evaluation_details"]["missing_facts"] ) != 0 else 0 for result in results])}')
        print("=" * 80)
        
    else:
        print("No results to display.")

def main():
    """Main function to run the Q&A testing."""
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load dataset
    try:
        dataset = load_dataset(DATASET_URL)
        print(f"Dataset loaded successfully. Found {len(dataset.get('qa_pairs', []))} Q&A pairs.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get random Q&A pairs
    qa_pairs = get_random_qa_pairs(dataset, NUM_PAIRS)
    print(f"\nSelected {len(qa_pairs)} random Q&A pairs for testing.\n")
    
    evaluate_with_fact_checking(client, MODEL_NAME, qa_pairs)


if __name__ == "__main__":
    main()





