"""
LLM Client Module
Handles Gemini (primary) and Groq (fallback) integration for query understanding
and recommendation ranking.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client with provider auto-selection (Gemini first, Groq fallback)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        env_model = model or os.getenv("LLM_MODEL")
        gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            candidate_model = env_model or "gemini-1.5-flash"
            if not candidate_model.lower().startswith("gemini"):
                logger.warning(
                    "LLM_MODEL '%s' is not a Gemini model; falling back to gemini-1.5-flash",
                    candidate_model,
                )
                candidate_model = "gemini-1.5-flash"
            self.model = candidate_model
            logger.info("LLM client initialized with Gemini model: %s", self.model)
            return

        groq_key = api_key or os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq  # type: ignore
            except Exception as e:
                raise ValueError(f"GROQ_API_KEY found but groq SDK import failed: {e}") from e

            self.provider = "groq"
            self.api_key = groq_key
            candidate_model = env_model or "llama-3.1-70b-versatile"
            if candidate_model.lower().startswith("gemini"):
                logger.warning(
                    "LLM_MODEL '%s' is not a Groq model; falling back to llama-3.1-70b-versatile",
                    candidate_model,
                )
                candidate_model = "llama-3.1-70b-versatile"
            self.model = candidate_model
            self.client = Groq(api_key=self.api_key)
            logger.info("LLM client initialized with Groq model: %s", self.model)
            return

        raise ValueError("No LLM API key found. Set GEMINI_API_KEY (preferred) or GROQ_API_KEY.")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from the selected provider."""
        try:
            if self.provider == "gemini":
                return self._generate_gemini(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return self._generate_groq(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("Error generating text: %s", e)
            return ""

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(self.temperature if temperature is None else temperature),
                "maxOutputTokens": int(self.max_tokens if max_tokens is None else max_tokens),
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = (candidates[0].get("content", {}) or {}).get("parts", []) or []
        text_chunks = [str(p.get("text", "")) for p in parts if p.get("text")]
        return "".join(text_chunks).strip()

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent (technical vs behavioral skills)."""
        system_prompt = """You are an expert SHL assessment consultant helping hiring managers find the right assessments for their job roles.

SHL Assessment Test Types:
- A: Ability & Aptitude - cognitive, numerical, verbal, logical reasoning tests
- B: Biodata & Situational Judgement - real world judgment and decision making
- C: Competencies - workplace competency based assessments
- D: Development & 360 - development and feedback assessments
- E: Assessment Exercises - practical job simulation exercises
- K: Knowledge & Skills - technical/domain specific knowledge tests (Java, Python, SQL etc.)
- P: Personality & Behavior - personality traits, behavioral styles, workplace preferences
- S: Simulations - job realistic simulations

Your job is to:
1. Analyze the query or job description
2. Extract hard skills (technical, domain specific) and soft skills (behavioral, personality)
3. Identify the most relevant SHL test types needed
4. Determine the balance between technical and behavioral assessments
5. Extract specific keywords to search the SHL catalog effectively

IMPORTANT RULES:
- If query mentions specific technical skills (Java, Python, SQL etc.), always include test type K
- If query mentions collaboration, communication, leadership, teamwork etc., always include test type P
- If query mentions reasoning, analytical, problem solving etc., always include test type A
- Always recommend a BALANCED mix of test types when query spans multiple domains
- Never recommend only one test type if the query clearly spans multiple domains

Return your analysis in this EXACT format without any additional text:
TECHNICAL_WEIGHT: <0.0 to 1.0>
BEHAVIORAL_WEIGHT: <0.0 to 1.0>
PRIMARY_SKILLS: <comma-separated list of specific skills extracted from query>
REQUIRED_TEST_TYPES: <comma-separated list of relevant test type letters e.g. K,P,A>
SEARCH_KEYWORDS: <comma-separated list of keywords to search SHL catalog>
MIN_TECHNICAL_ASSESSMENTS: <minimum number of technical assessments to include, 1-5>
MIN_BEHAVIORAL_ASSESSMENTS: <minimum number of behavioral assessments to include, 1-5>
REASONING: <brief explanation of why these test types and balance were chosen>"""

        prompt = f"""Analyze this job requirement or query and recommend appropriate SHL assessment types:

    "{query}"

Consider:
1. What specific technical skills or domain knowledge is required?
2. What behavioral or personality traits are important for this role?
3. What is the seniority level (junior, mid, senior) and how does it affect assessment choice?
4. Does this role require collaboration with teams or stakeholders?
5. Are there any cognitive or reasoning requirements mentioned?

Provide a balanced assessment recommendation that covers ALL skill dimensions mentioned in the query."""

        try:
            response = self.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=400,
            )
            result: Dict[str, Any] = {
                "technical_weight": 0.6,
                "behavioral_weight": 0.4,
                "primary_skills": [],
                "reasoning": "",
            }

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("TECHNICAL_WEIGHT:"):
                    try:
                        result["technical_weight"] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.startswith("BEHAVIORAL_WEIGHT:"):
                    try:
                        result["behavioral_weight"] = float(line.split(":")[1].strip())
                    except Exception:
                        pass
                elif line.startswith("PRIMARY_SKILLS:"):
                    skills_str = line.split(":", 1)[1].strip()
                    result["primary_skills"] = [s.strip() for s in skills_str.split(",") if s.strip()]
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()

            total = float(result["technical_weight"]) + float(result["behavioral_weight"])
            if total > 0:
                result["technical_weight"] /= total
                result["behavioral_weight"] /= total

            logger.info(
                "Query classified: %.2f technical, %.2f behavioral",
                result["technical_weight"],
                result["behavioral_weight"],
            )
            return result
        except Exception as e:
            logger.error("Error classifying query: %s", e)
            return {
                "technical_weight": 0.6,
                "behavioral_weight": 0.4,
                "primary_skills": [],
                "reasoning": "Error in classification",
            }

    def expand_query(self, query: str) -> str:
        system_prompt = """You are an expert SHL assessment consultant with deep knowledge of the SHL product catalog and psychometric assessments.

Your task is to expand job queries or job descriptions to improve retrieval of relevant SHL assessments from the catalog.

SHL Assessment Categories to consider during expansion:
- Ability & Aptitude (A): numerical reasoning, verbal reasoning, logical reasoning, cognitive ability, critical thinking
- Biodata & Situational Judgement (B): judgment, decision making, situational awareness, real world scenarios
- Competencies (C): workplace competencies, leadership competencies, management skills
- Development & 360 (D): feedback, development planning, self assessment, 360 review
- Assessment Exercises (E): role play, in-tray exercises, group exercises, presentations
- Knowledge & Skills (K): technical knowledge, domain expertise, programming languages, tools, frameworks
- Personality & Behavior (P): personality traits, behavioral styles, work preferences, motivation, values
- Simulations (S): job simulations, realistic job previews, work samples

Expansion Rules:
1. TECHNICAL SKILLS: Expand specific technologies to include related ones
   - e.g., "Java" → "Java, object oriented programming, software development, coding, programming"
   - e.g., "Python" → "Python, data analysis, scripting, automation, programming"
   - e.g., "SQL" → "SQL, database, data querying, relational database, data management"

2. SOFT SKILLS: Expand behavioral terms to include SHL relevant personality concepts
   - e.g., "collaboration" → "collaboration, teamwork, interpersonal skills, stakeholder management, communication"
   - e.g., "leadership" → "leadership, people management, team management, decision making, influence"

3. ROLE CONTEXT: Add role specific assessment terms
   - e.g., "developer" → "developer, software engineer, technical skills, coding ability, problem solving"
   - e.g., "analyst" → "analyst, analytical thinking, numerical reasoning, data interpretation, critical thinking"
   - e.g., "manager" → "manager, leadership, team management, competencies, behavioral assessment"

4. SENIORITY CONTEXT: Add seniority relevant terms
   - e.g., "senior" → "senior, experienced, strategic thinking, leadership, mentoring"
   - e.g., "mid-level" → "mid-level, intermediate, independent, problem solving"
   - e.g., "entry level" → "entry level, graduate, potential, learning ability, aptitude"

5. ALWAYS include both technical AND behavioral expansion if query mentions both domains
6. NEVER remove original terms from the query
7. NEVER add irrelevant terms that are not related to the query
8. Keep expansion focused on terms that exist in SHL assessment catalog

Return your response in this EXACT format without any additional text:
EXPANDED_QUERY: <full expanded query with all related terms>
TECHNICAL_TERMS: <comma-separated list of technical terms added>
BEHAVIORAL_TERMS: <comma-separated list of behavioral terms added>
ROLE_TERMS: <comma-separated list of role specific terms added>
KEY_CONCEPTS: <comma-separated list of top 5 most important concepts for retrieval>"""

        prompt = f"""Expand this job requirement query to improve SHL assessment retrieval:

"{query}"

Consider:
1. What specific technical skills or tools are mentioned? Add related technologies and concepts.
2. What behavioral or soft skills are mentioned? Add related personality and competency terms.
3. What is the role type? Add role specific assessment terminology.
4. What seniority level is mentioned? Add appropriate level specific terms.
5. What SHL assessment categories (A, B, C, D, E, K, P, S) are most relevant? Add their related terms.

Expand comprehensively while staying relevant to the SHL assessment catalog."""

        try:
            expanded = self.generate(prompt, system_prompt=system_prompt, max_tokens=200)
            return expanded if expanded else query
        except Exception as e:
            logger.error("Error expanding query: %s", e)
            return query

    def rerank_assessments(self, query: str, assessments: List[Dict], top_k: int = 10) -> List[Dict]:
        """Re-rank assessments using LLM understanding."""
        if not assessments:
            return []

        rerank_input_limit = min(len(assessments), 50)
        assessment_list = []
        for i, assess in enumerate(assessments[:rerank_input_limit]):
            assessment_list.append(
                f"{i + 1}. {assess.get('assessment_name', 'Unknown')} "
                f"(Type: {assess.get('test_type', 'N/A')}) - "
                f"{assess.get('description', 'No description')[:100]}"
            )

        system_prompt = """You are an SHL assessment expert. Rank assessments by relevance to the job query.

Rules:
- Balance TECHNICAL (K, A) and BEHAVIORAL (P, C, B) assessments if query mentions both
- Prioritize specific skill matches (e.g., Java query → Java assessment first)
- Include mix of test types: never return all same test type if query spans multiple domains
- Prefer assessments whose duration/test type constraints explicitly match the query
- Return ONLY comma-separated numbers, nothing else

Example output: 3,1,7,2,5"""

        prompt = f"""Query: "{query}"

Assessments:
{chr(10).join(assessment_list)}

Return top {min(top_k, rerank_input_limit)} assessment numbers (comma-separated), balanced across skill types."""

        try:
            response = self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=100,
            )
            numbers = re.findall(r"\d+", response)
            ranked_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(assessments)]

            reranked: List[Dict] = []
            seen = set()
            for idx in ranked_indices[:top_k]:
                if idx < len(assessments) and idx not in seen:
                    reranked.append(assessments[idx])
                    seen.add(idx)

            for i, assess in enumerate(assessments):
                if i not in seen and len(reranked) < top_k:
                    reranked.append(assess)
                    seen.add(i)

            logger.info("Re-ranked %s assessments", len(reranked))
            return reranked[:top_k]
        except Exception as e:
            logger.error("Error re-ranking assessments: %s", e)
            return assessments[:top_k]

    def generate_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """Generate explanation for recommendations."""
        assessment_names = [r.get("assessment_name", "Unknown") for r in recommendations[:5]]

        system_prompt = """You are an expert at explaining assessment recommendations.
Provide a brief, clear explanation of why these assessments are relevant to the query.
Keep it concise (2-3 sentences)."""

        prompt = f"""Query: "{query}"

Recommended Assessments:
{chr(10).join(f"- {name}" for name in assessment_names)}

Explain why these assessments are relevant to the query."""

        try:
            explanation = self.generate(prompt, system_prompt=system_prompt, max_tokens=150)
            return explanation if explanation else "These assessments match your requirements."
        except Exception as e:
            logger.error("Error generating explanation: %s", e)
            return "These assessments match your requirements."


def create_llm_client(api_key: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Create LLM client instance."""
    return LLMClient(api_key=api_key, model=model)


if __name__ == "__main__":
    try:
        client = LLMClient()
        test_query = "Looking for Java developer with strong teamwork skills"
        result = client.classify_query_intent(test_query)
        print(f"Query: {test_query}")
        print(f"Provider: {client.provider}")
        print(f"Model: {client.model}")
        print(f"Technical weight: {result['technical_weight']:.2f}")
        print(f"Behavioral weight: {result['behavioral_weight']:.2f}")
        print(f"Primary skills: {result['primary_skills']}")
        print(f"Reasoning: {result['reasoning']}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Set GEMINI_API_KEY in .env file (preferred), or GROQ_API_KEY as fallback.")
