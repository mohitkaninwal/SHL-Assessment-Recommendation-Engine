"""
LLM Client Module
Handles Groq API integration for query understanding and recommendation ranking
"""

import logging
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """Groq LLM client for query understanding and ranking"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize Groq LLM client
        
        Args:
            api_key: Groq API key (defaults to env var)
            model: Model name (defaults to env var or llama-3.1-70b-versatile)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = model or os.getenv('LLM_MODEL', 'llama-3.1-70b-versatile')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq LLM client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Groq LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def classify_query_intent(self, query: str) -> Dict[str, any]:
        """
        Classify query intent (technical vs behavioral skills)
        
        Args:
            query: User query
            
        Returns:
            Dictionary with classification results
        """
        system_prompt = """You are an expert at analyzing job requirements and skill assessments.
Classify the given query into skill categories and determine the balance needed between:
- Technical/Knowledge skills (hard skills)
- Behavioral/Personality skills (soft skills)

Return your analysis in this exact format:
TECHNICAL_WEIGHT: <0.0 to 1.0>
BEHAVIORAL_WEIGHT: <0.0 to 1.0>
PRIMARY_SKILLS: <comma-separated list>
REASONING: <brief explanation>"""
        
        prompt = f"""Analyze this job requirement or query:

"{query}"

Classify the skills needed and determine the appropriate balance between technical and behavioral assessments."""
        
        try:
            response = self.generate(prompt, system_prompt=system_prompt)
            
            # Parse response
            result = {
                'technical_weight': 0.6,  # Default
                'behavioral_weight': 0.4,
                'primary_skills': [],
                'reasoning': ''
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('TECHNICAL_WEIGHT:'):
                    try:
                        result['technical_weight'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('BEHAVIORAL_WEIGHT:'):
                    try:
                        result['behavioral_weight'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('PRIMARY_SKILLS:'):
                    skills_str = line.split(':', 1)[1].strip()
                    result['primary_skills'] = [s.strip() for s in skills_str.split(',')]
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()
            
            # Normalize weights
            total = result['technical_weight'] + result['behavioral_weight']
            if total > 0:
                result['technical_weight'] /= total
                result['behavioral_weight'] /= total
            
            logger.info(f"Query classified: {result['technical_weight']:.2f} technical, {result['behavioral_weight']:.2f} behavioral")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return {
                'technical_weight': 0.6,
                'behavioral_weight': 0.4,
                'primary_skills': [],
                'reasoning': 'Error in classification'
            }
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms and concepts
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        system_prompt = """You are an expert at understanding job requirements and skill assessments.
Expand the given query by adding related terms, synonyms, and relevant concepts that would help find appropriate assessments.
Keep the expansion concise and relevant."""
        
        prompt = f"""Expand this query with related terms and concepts:

"{query}"

Provide an expanded version that includes synonyms and related skills."""
        
        try:
            expanded = self.generate(prompt, system_prompt=system_prompt, max_tokens=200)
            return expanded if expanded else query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query
    
    def rerank_assessments(
        self,
        query: str,
        assessments: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank assessments using LLM understanding
        
        Args:
            query: User query
            assessments: List of retrieved assessments
            top_k: Number of top assessments to return
            
        Returns:
            Re-ranked list of assessments
        """
        if not assessments:
            return []
        
        # Prepare assessment list for LLM
        assessment_list = []
        for i, assess in enumerate(assessments[:20]):  # Limit to top 20 for LLM
            assessment_list.append(
                f"{i+1}. {assess.get('assessment_name', 'Unknown')} "
                f"(Type: {assess.get('test_type', 'N/A')}) - "
                f"{assess.get('description', 'No description')[:100]}"
            )
        
        system_prompt = """You are an expert at matching job requirements with appropriate skill assessments.
Given a query and a list of assessments, rank them by relevance.
Consider both technical fit and the balance of hard/soft skills needed.

Return ONLY the numbers of the top assessments in order, comma-separated.
Example: 3,1,7,2,5"""
        
        prompt = f"""Query: "{query}"

Assessments:
{chr(10).join(assessment_list)}

Rank these assessments by relevance to the query. Return only the numbers (comma-separated) of the top {min(top_k, len(assessments))} most relevant assessments."""
        
        try:
            response = self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent ranking
                max_tokens=100
            )
            
            # Parse ranking
            try:
                # Extract numbers from response
                import re
                numbers = re.findall(r'\d+', response)
                ranked_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(assessments)]
                
                # Reorder assessments
                reranked = []
                seen = set()
                
                for idx in ranked_indices[:top_k]:
                    if idx < len(assessments) and idx not in seen:
                        reranked.append(assessments[idx])
                        seen.add(idx)
                
                # Add remaining assessments if needed
                for i, assess in enumerate(assessments):
                    if i not in seen and len(reranked) < top_k:
                        reranked.append(assess)
                        seen.add(i)
                
                logger.info(f"Re-ranked {len(reranked)} assessments")
                return reranked[:top_k]
                
            except Exception as e:
                logger.warning(f"Error parsing ranking, returning original order: {e}")
                return assessments[:top_k]
                
        except Exception as e:
            logger.error(f"Error re-ranking assessments: {e}")
            return assessments[:top_k]
    
    def generate_explanation(
        self,
        query: str,
        recommendations: List[Dict]
    ) -> str:
        """
        Generate explanation for recommendations
        
        Args:
            query: User query
            recommendations: List of recommended assessments
            
        Returns:
            Explanation text
        """
        assessment_names = [r.get('assessment_name', 'Unknown') for r in recommendations[:5]]
        
        system_prompt = """You are an expert at explaining assessment recommendations.
Provide a brief, clear explanation of why these assessments are relevant to the query.
Keep it concise (2-3 sentences)."""
        
        prompt = f"""Query: "{query}"

Recommended Assessments:
{chr(10).join(f"- {name}" for name in assessment_names)}

Explain why these assessments are relevant to the query."""
        
        try:
            explanation = self.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=150
            )
            return explanation if explanation else "These assessments match your requirements."
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "These assessments match your requirements."


def create_llm_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """
    Create LLM client instance
    
    Args:
        api_key: Optional API key override
        model: Optional model override
        
    Returns:
        LLMClient instance
    """
    return LLMClient(api_key=api_key, model=model)


if __name__ == "__main__":
    # Test LLM client
    try:
        client = LLMClient()
        
        # Test query classification
        test_query = "Looking for Java developer with strong teamwork skills"
        result = client.classify_query_intent(test_query)
        
        print(f"Query: {test_query}")
        print(f"Technical weight: {result['technical_weight']:.2f}")
        print(f"Behavioral weight: {result['behavioral_weight']:.2f}")
        print(f"Primary skills: {result['primary_skills']}")
        print(f"Reasoning: {result['reasoning']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Set GROQ_API_KEY in .env file")
        print("Get your key from: https://console.groq.com/")
