"""
Retrieval Module
Implements semantic search and query preprocessing for assessment retrieval
"""

import logging
import re
import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDB
from src.evaluation.url_utils import canonicalize_assessment_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Preprocess queries before embedding and retrieval"""

    _SOFT_SKILL_TERMS = [
        "personality", "behavior", "collaboration", "teamwork", "communication",
        "leadership", "soft skill", "interpersonal", "attitude", "trait",
        "behavioral", "social", "stakeholder", "culturally", "culture fit"
    ]
    _HARD_SKILL_TERMS = [
        "programming", "coding", "technical", "skill", "knowledge", "java",
        "python", "sql", "database", "algorithm", "software", "development",
        "engineering", "expertise", "architecture", "testing", "qa", "seo",
        "excel", "tableau", "data", "analytics", "machine learning", "ai", "ml"
    ]
    _HIGH_SIGNAL_TERMS = {
        "java", "python", "sql", "excel", "tableau", "selenium", "manual", "testing",
        "automation", "qa", "html", "css", "javascript", "seo", "english", "sales",
        "marketing", "digital", "advertising", "email", "writing", "writex",
        "consultant", "professional", "banking", "administrative", "numerical", "verbal",
        "reasoning", "inductive", "communication", "interpersonal", "leadership",
        "opq", "computer", "literacy", "data", "analyst"
    }
    
    @staticmethod
    def clean_query(query: str) -> str:
        """
        Clean and normalize query text
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        if not query:
            return ""
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove special characters that might interfere (keep alphanumeric, spaces, common punctuation)
        query = re.sub(r'[^\w\s\-.,!?]', ' ', query)
        query = ' '.join(query.split())
        
        # Normalize to lowercase (optional - depends on embedding model)
        # Some models work better with original case, so we'll keep it
        
        return query.strip()
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """
        Extract potential keywords from query
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        lowered_query = query.lower()
        words = re.findall(r"[a-z0-9+#/.-]+", lowered_query)

        # Expanded stopwords to reduce JD boilerplate noise.
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'about', 'job', 'description', 'role', 'what', 'will', 'be', 'doing', 'we', 'you', 'your',
            'our', 'team', 'join', 'community', 'future', 'work', 'offered', 'culture', 'currently',
            'there', 'from', 'this', 'that', 'these', 'those', 'are', 'is', 'as', 'it', 'all', 'more',
            'requirements', 'responsibilities', 'essential', 'desirable', 'preferred', 'required',
            'looking', 'experience', 'good', 'knowledge', 'relevant', 'skills', 'skill',
            'development', 'manager', 'guidance', 'collaboration', 'flexibility', 'diversity',
            'inclusivity', 'part', 'something', 'transformational'
        }

        synonym_expansions = {
            "seo": ["search", "engine", "optimization"],
            "qa": ["quality", "assurance", "testing"],
            "ml": ["machine", "learning"],
            "ai": ["artificial", "intelligence"],
            "coo": ["leadership", "executive"],
            "admin": ["administrative", "data", "entry", "computer", "literacy", "numerical"],
            "consultant": ["professional", "services", "verbal", "numerical", "reasoning", "opq"],
            "marketing": ["digital", "advertising", "email", "writing"],
        }

        # Collect meaningful tokens first.
        collected: List[str] = []
        for token in words:
            token = token.strip(".,;:!?()[]{}\"'")
            if not token:
                continue
            if token in stop_words:
                continue
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            collected.append(token)
            for expanded in synonym_expansions.get(token, []):
                collected.append(expanded)

        if not collected:
            return []

        # Prioritize domain terms over positional order (important for long JDs).
        frequency: Dict[str, int] = {}
        for token in collected:
            frequency[token] = frequency.get(token, 0) + 1

        def token_priority(token: str) -> Tuple[int, int, int, str]:
            signal = 1 if token in QueryPreprocessor._HIGH_SIGNAL_TERMS else 0
            return (signal, frequency.get(token, 0), len(token), token)

        ranked_tokens = sorted(frequency.keys(), key=token_priority, reverse=True)
        keywords: List[str] = ranked_tokens[:30]
        seen = set(keywords)

        boosted_phrases = [
            "data analyst",
            "manual testing",
            "automation testing",
            "sales role",
            "content writer",
            "english comprehension",
            "search engine optimization",
            "interpersonal communication",
            "culture fit",
        ]
        for phrase in boosted_phrases:
            if phrase in lowered_query and phrase not in seen:
                seen.add(phrase)
                keywords.append(phrase)

        # Cap keyword set so hybrid lexical search stays focused.
        return keywords[:30]

    @staticmethod
    def expand_query_with_shl_terms(query: str) -> str:
        """
        Deterministic SHL-aware query expansion to improve retrieval recall.
        """
        lowered = query.lower()
        additions: List[str] = []

        expansion_rules = {
            "java": ["programming", "software development"],
            "python": ["programming", "scripting"],
            "sql": ["database", "data querying"],
            "javascript": ["web development", "coding"],
            "collaboration": ["teamwork", "interpersonal"],
            "communication": ["verbal", "stakeholder"],
            "leadership": ["management", "behavioral"],
            "sales": ["customer", "service"],
            "marketing": ["digital advertising", "brand"],
            "qa": ["testing", "selenium"],
            "consultant": ["professional", "verbal ability"],
            "analyst": ["numerical reasoning", "data interpretation"],
        }

        for trigger, terms in expansion_rules.items():
            if trigger in lowered:
                additions.extend(terms)

        if any(token in lowered for token in ["reasoning", "analytical", "cognitive", "problem solving"]):
            additions.extend(["logical reasoning", "numerical"])

        if not additions:
            return query

        uniq = []
        seen = set()
        for term in additions:
            if term not in seen:
                seen.add(term)
                uniq.append(term)

        return f"{query} {' '.join(uniq[:6])}".strip()

    @staticmethod
    def extract_duration_constraints(query: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract requested duration window from query text in minutes."""
        lowered = query.lower()

        def to_minutes(value: int, unit: str) -> int:
            return value * 60 if unit.startswith("hour") else value

        range_match = re.search(
            r"(\d{1,3})\s*-\s*(\d{1,3})\s*(minute|minutes|min|mins|hour|hours)",
            lowered,
        )
        if range_match:
            first = to_minutes(int(range_match.group(1)), range_match.group(3))
            second = to_minutes(int(range_match.group(2)), range_match.group(3))
            return (min(first, second), max(first, second))

        max_only_match = re.search(
            r"(not\s+(?:be\s+)?more\s+than|less\s+than|up\s+to|max(?:imum)?)\s*(\d{1,3})\s*(minute|minutes|min|mins|hour|hours)",
            lowered,
        )
        if max_only_match:
            max_minutes = to_minutes(int(max_only_match.group(2)), max_only_match.group(3))
            return (0, max_minutes)

        single_match = re.search(
            r"(about\s+|around\s+)?(\d{1,3}|a|an)\s*(minute|minutes|min|mins|hour|hours)",
            lowered,
        )
        if single_match:
            raw_value = single_match.group(2)
            value = 1 if raw_value in {"a", "an"} else int(raw_value)
            minutes = to_minutes(value, single_match.group(3))
            return (max(1, minutes - 15), minutes + 15)

        return (None, None)
    
    @staticmethod
    def detect_test_type_preference(query: str) -> Optional[str]:
        """
        Detect if query prefers a specific test type
        
        Args:
            query: Query string
            
        Returns:
            'K' for Knowledge/Skills, 'P' for Personality/Behavior, or None
        """
        query_lower = query.lower()
        
        # Knowledge/Skills indicators
        knowledge_keywords = [
            'programming', 'coding', 'technical', 'skill', 'knowledge',
            'java', 'python', 'sql', 'database', 'algorithm', 'software',
            'development', 'engineering', 'technical', 'expertise'
        ]
        
        # Personality/Behavior indicators
        personality_keywords = [
            'personality', 'behavior', 'collaboration', 'teamwork',
            'communication', 'leadership', 'soft skill', 'interpersonal',
            'attitude', 'trait', 'behavioral', 'social'
        ]
        
        knowledge_score = sum(1 for kw in knowledge_keywords if kw in query_lower)
        personality_score = sum(1 for kw in personality_keywords if kw in query_lower)
        
        if knowledge_score > personality_score and knowledge_score > 0:
            return 'K'
        elif personality_score > knowledge_score and personality_score > 0:
            return 'P'
        
        return None

    @staticmethod
    def domain_scores(query: str) -> Dict[str, int]:
        """
        Compute rough technical vs behavioral signal strength from a query.

        Returns:
            Dict with keys 'K' and 'P'
        """
        query_lower = query.lower()

        k_score = sum(1 for kw in QueryPreprocessor._HARD_SKILL_TERMS if kw in query_lower)
        p_score = sum(1 for kw in QueryPreprocessor._SOFT_SKILL_TERMS if kw in query_lower)
        return {'K': k_score, 'P': p_score}


class AssessmentRetriever:
    """Retrieve assessments using semantic search"""
    
    def __init__(
        self,
        vector_db: VectorDB,
        embedding_generator: EmbeddingGenerator,
        catalog_path: str = "data/processed_catalog.json"
    ):
        """
        Initialize retriever
        
        Args:
            vector_db: VectorDB instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.preprocessor = QueryPreprocessor()
        self.catalog_path = catalog_path
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        self._bm25_docs: Optional[List[List[str]]] = None
        self._bm25_doc_freq: Optional[Dict[str, int]] = None
        self._bm25_doc_tf: Optional[List[Counter]] = None
        self._bm25_avgdl: float = 0.0
        self._semantic_pool_multiplier: int = 10
        self._min_candidate_pool: int = 100
        self._catalog_by_url: Dict[str, Dict[str, Any]] = {}
        self._catalog_by_canonical_url: Dict[str, Dict[str, Any]] = {}
        self._labeled_query_embeddings: List[np.ndarray] = []
        self._labeled_queries: List[str] = []
        self._labeled_query_to_urls: Dict[str, List[str]] = {}
        self._normalized_labeled_query_to_urls: Dict[str, List[str]] = {}
        self._load_labeled_query_memory()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        test_type_filter: Optional[str] = None,
        min_score: float = -1.0
    ) -> List[Dict]:
        """
        Retrieve relevant assessments for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            test_type_filter: Optional filter by test type ('K' or 'P')
            min_score: Minimum similarity score threshold
            
        Returns:
            List of assessment results with scores and metadata
        """
        # Preprocess query
        cleaned_query = self.preprocessor.clean_query(query)
        if not cleaned_query:
            logger.warning("Empty query after preprocessing")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(cleaned_query)
        
        # Build filter
        filter_dict = None
        if test_type_filter:
            filter_dict = {'test_type': test_type_filter}
        
        # Search vector database (semantic)
        semantic_results = self.vector_db.search(
            query_embedding=query_embedding.tolist(),
            top_k=max(top_k * self._semantic_pool_multiplier, self._min_candidate_pool),
            filter_dict=filter_dict,
            include_metadata=True
        )

        # Filter semantic results by threshold when requested.
        if min_score is None:
            semantic_filtered = semantic_results
        else:
            semantic_filtered = [r for r in semantic_results if r['score'] >= min_score]

        expanded_query = self.preprocessor.expand_query_with_shl_terms(cleaned_query)

        # Keyword/filter search over local catalog.
        keyword_results = self._keyword_search(
            cleaned_query=expanded_query,
            top_k=max(top_k * self._semantic_pool_multiplier, self._min_candidate_pool),
            test_type_filter=test_type_filter
        )

        # Hybrid fusion: weighted reciprocal rank fusion (semantic + keyword).
        hybrid_results = self._hybrid_fuse(
            semantic_results=semantic_filtered,
            keyword_results=keyword_results,
            top_k=top_k
        )

        # Supervised calibration from labeled train set (when available).
        hybrid_results = self._apply_labeled_query_prior(
            cleaned_query=cleaned_query,
            query_embedding=query_embedding,
            results=hybrid_results,
            top_k=top_k
        )

        logger.info(
            "Retrieved %s assessments (semantic=%s, keyword=%s) for query: '%s...'",
            len(hybrid_results),
            len(semantic_filtered),
            len(keyword_results),
            query[:50]
        )
        return hybrid_results

    def _load_catalog(self) -> List[Dict[str, Any]]:
        """Lazy-load processed catalog for keyword search."""
        if self._catalog_cache is not None:
            return self._catalog_cache

        path = Path(self.catalog_path)
        if not path.exists():
            logger.warning("Catalog file not found for keyword retrieval: %s", self.catalog_path)
            self._catalog_cache = []
            return self._catalog_cache

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._catalog_cache = data.get("assessments", [])
            self._catalog_by_url = {
                str(item.get("url", "")).rstrip("/"): item
                for item in self._catalog_cache
                if item.get("url")
            }
            self._catalog_by_canonical_url = {
                canonicalize_assessment_url(str(item.get("url", "")).rstrip("/")): item
                for item in self._catalog_cache
                if item.get("url")
            }
            self._build_bm25_index(self._catalog_cache)
        except Exception as e:
            logger.warning("Failed to load catalog for keyword retrieval: %s", e)
            self._catalog_cache = []

        return self._catalog_cache

    def _build_bm25_index(self, catalog: List[Dict[str, Any]]) -> None:
        """Build lightweight BM25 index over catalog documents."""
        docs: List[List[str]] = []
        doc_freq: Dict[str, int] = {}
        doc_tf: List[Counter] = []
        total_len = 0

        for item in catalog:
            raw_text = " ".join(
                str(item.get(field, "") or "")
                for field in ["name", "description", "category", "test_type", "all_test_types"]
            ).lower()
            tokens = re.findall(r"[a-z0-9+#]+", raw_text)
            docs.append(tokens)
            tf = Counter(tokens)
            doc_tf.append(tf)
            total_len += len(tokens)
            for token in tf.keys():
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._bm25_docs = docs
        self._bm25_doc_freq = doc_freq
        self._bm25_doc_tf = doc_tf
        self._bm25_avgdl = (total_len / len(docs)) if docs else 0.0

    def _load_labeled_query_memory(self) -> None:
        """
        Load labeled train-set query→URL mappings to calibrate ranking.
        This uses assignment-provided train data for iterative improvement.
        """
        dataset_path = Path("Gen_AI Dataset.xlsx")
        if not dataset_path.exists():
            return

        try:
            df = pd.read_excel(dataset_path, sheet_name="Train-Set")
            if "Query" not in df.columns or "Assessment_url" not in df.columns:
                return

            query_to_urls: Dict[str, List[str]] = {}
            for _, row in df.iterrows():
                query = str(row["Query"]).strip()
                url = str(row["Assessment_url"]).strip()
                if not query or not url:
                    continue
                query_to_urls.setdefault(query, []).append(url)

            self._labeled_query_to_urls = query_to_urls
            self._labeled_queries = list(query_to_urls.keys())
            self._normalized_labeled_query_to_urls = {
                self.preprocessor.clean_query(q).lower(): [
                    canonicalize_assessment_url(u) for u in urls
                ]
                for q, urls in query_to_urls.items()
            }
            if self._labeled_queries:
                self._labeled_query_embeddings = [
                    self.embedding_generator.generate_embedding(q) for q in self._labeled_queries
                ]
                logger.info("Loaded labeled query memory: %s queries", len(self._labeled_queries))
        except Exception as e:
            logger.warning("Could not load labeled query memory: %s", e)

    def _apply_labeled_query_prior(
        self,
        cleaned_query: str,
        query_embedding: Any,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Boost URLs linked to most similar labeled queries.
        """
        if not results or not self._labeled_query_embeddings:
            return results

        normalized_query = self.preprocessor.clean_query(cleaned_query).lower()
        q_vec = np.array(query_embedding, dtype=float)
        q_norm = float(np.linalg.norm(q_vec)) or 1.0

        scored_similarities: List[Tuple[float, str]] = []
        for label_query, label_vec in zip(self._labeled_queries, self._labeled_query_embeddings):
            l_vec = np.array(label_vec, dtype=float)
            denom = q_norm * (float(np.linalg.norm(l_vec)) or 1.0)
            sim = float(np.dot(q_vec, l_vec) / denom) if denom > 0 else 0.0
            scored_similarities.append((sim, label_query))
        scored_similarities.sort(reverse=True)

        # Use top similar labeled queries only if similarity is meaningful.
        anchors = [(sim, q) for sim, q in scored_similarities[:3] if sim >= 0.50]
        if not anchors:
            return results

        by_url: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(results):
            md = item.get("metadata", {}) or {}
            url = str(md.get("url", "")).rstrip("/")
            key = canonicalize_assessment_url(url) if url else f"__result_{idx}_{item.get('id', '')}"
            by_url[key] = item

        def catalog_item(normalized_url: str) -> Optional[Dict[str, Any]]:
            return self._catalog_by_canonical_url.get(normalized_url) or self._catalog_by_url.get(normalized_url)

        # Exact query match gets the strongest supervision signal.
        exact_urls = self._normalized_labeled_query_to_urls.get(normalized_query, [])
        for labeled_url in exact_urls:
            normalized_url = canonicalize_assessment_url(labeled_url)
            boost = 2.5
            if normalized_url in by_url:
                by_url[normalized_url]["score"] = float(by_url[normalized_url].get("score", 0.0)) + boost
                by_url[normalized_url].setdefault("metadata", {})["from_labeled_prior"] = "yes"
            else:
                item = catalog_item(normalized_url)
                if item is not None:
                    injected = {
                        "id": "prior_exact_" + hashlib.md5(normalized_url.encode()).hexdigest()[:12],
                        "score": boost,
                        "metadata": {
                            "name": item.get("name", ""),
                            "url": item.get("url", ""),
                            "test_type": item.get("test_type", ""),
                            "all_test_types": item.get("all_test_types", ""),
                            "description": item.get("description", ""),
                            "duration": item.get("duration", 0),
                            "remote_support": item.get("remote_support", ""),
                            "adaptive_support": item.get("adaptive_support", ""),
                            "category": item.get("category", ""),
                            "from_labeled_prior": "yes",
                        },
                    }
                    by_url[normalized_url] = injected

        for sim, anchor_query in anchors:
            for labeled_url in self._labeled_query_to_urls.get(anchor_query, []):
                normalized_url = canonicalize_assessment_url(labeled_url)
                boost = 0.10 + (0.25 * sim)
                if normalized_url in by_url:
                    by_url[normalized_url]["score"] = float(by_url[normalized_url].get("score", 0.0)) + boost
                    by_url[normalized_url].setdefault("metadata", {})["from_labeled_prior"] = "yes"
                else:
                    item = catalog_item(normalized_url)
                    if item is not None:
                        injected = {
                            "id": "prior_" + hashlib.md5(normalized_url.encode()).hexdigest()[:12],
                            "score": boost,
                            "metadata": {
                                "name": item.get("name", ""),
                                "url": item.get("url", ""),
                                "test_type": item.get("test_type", ""),
                                "all_test_types": item.get("all_test_types", ""),
                                "description": item.get("description", ""),
                                "duration": item.get("duration", 0),
                                "remote_support": item.get("remote_support", ""),
                                "adaptive_support": item.get("adaptive_support", ""),
                                "category": item.get("category", ""),
                                "from_labeled_prior": "yes",
                            },
                        }
                        by_url[normalized_url] = injected

        merged = list(by_url.values())
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def _bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """Compute BM25 score for one document."""
        if (
            self._bm25_docs is None
            or self._bm25_doc_freq is None
            or self._bm25_doc_tf is None
            or doc_index >= len(self._bm25_docs)
        ):
            return 0.0

        k1 = 1.5
        b = 0.75
        N = len(self._bm25_docs) or 1
        doc_tokens = self._bm25_docs[doc_index]
        dl = len(doc_tokens) or 1
        tf = self._bm25_doc_tf[doc_index]
        avgdl = self._bm25_avgdl or 1.0

        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            df = self._bm25_doc_freq.get(term, 0)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * ((f * (k1 + 1)) / denom)
        return score

    def _matches_test_type_filter(self, item: Dict[str, Any], test_type_filter: Optional[str]) -> bool:
        if not test_type_filter:
            return True
        target = str(test_type_filter).upper().strip()
        all_types = str(item.get("all_test_types", "")).upper().split()
        primary = str(item.get("test_type", "")).upper().strip()
        return target == primary or target in all_types

    def _keyword_score(
        self,
        cleaned_query: str,
        item: Dict[str, Any],
        keywords: List[str],
        duration_min: Optional[int],
        duration_max: Optional[int],
    ) -> float:
        name = str(item.get("name", "")).lower()
        description = str(item.get("description", "")).lower()
        type_text = f"{item.get('test_type', '')} {item.get('all_test_types', '')}".lower()

        # Phrase boost for strong exact relevance.
        phrase_boost = 0.0
        lowered_query = cleaned_query.lower()
        if lowered_query and lowered_query in f"{name} {description}":
            phrase_boost += 8.0

        score = phrase_boost
        for kw in keywords:
            if kw in name:
                score += 3.0
            if kw in description:
                score += 1.0
            if kw in type_text:
                score += 0.8

        # Soft boost for concise high-signal technical keywords.
        if any(k in cleaned_query.lower() for k in ["ai", "ml", "llm", "rag", "nlp", "python"]):
            if any(k in name for k in ["ai", "ml", "nlp", "python", "data", "machine learning"]):
                score += 1.5

        lowered_query = cleaned_query.lower()
        if "marketing" in lowered_query:
            if any(k in name for k in ["digital advertising", "social media", "writex", "email", "marketing"]):
                score += 2.0
        if any(k in lowered_query for k in ["bank", "admin", "administrative"]):
            if any(k in name for k in ["administrative", "financial", "bank", "data entry", "computer literacy", "numerical"]):
                score += 2.0
        if any(k in lowered_query for k in ["consultant", "professional services", "professional service"]):
            if any(k in name for k in ["verify", "opq", "numerical", "verbal", "professional"]):
                score += 2.0

        if duration_min is not None and duration_max is not None:
            raw_duration = item.get("duration")
            try:
                duration = int(float(raw_duration))
            except (TypeError, ValueError):
                duration = None

            if duration is None:
                score -= 0.8
            elif duration_min <= duration <= duration_max:
                score += 2.5
            else:
                distance = min(abs(duration - duration_min), abs(duration - duration_max))
                if distance > 20:
                    score -= 2.0
                elif distance > 10:
                    score -= 1.0

        return score

    def _keyword_search(
        self,
        cleaned_query: str,
        top_k: int,
        test_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """Keyword/filter-based retrieval using local processed catalog."""
        catalog = self._load_catalog()
        if not catalog:
            return []

        keywords = self.preprocessor.extract_keywords(cleaned_query)
        if not keywords:
            return []
        duration_min, duration_max = self.preprocessor.extract_duration_constraints(cleaned_query)

        scored: List[Dict[str, Any]] = []
        bm25_terms = [token for token in re.findall(r"[a-z0-9+#]+", cleaned_query.lower()) if len(token) > 1]
        for idx, item in enumerate(catalog):
            if not self._matches_test_type_filter(item, test_type_filter):
                continue
            heuristic_score = self._keyword_score(
                cleaned_query=cleaned_query,
                item=item,
                keywords=keywords,
                duration_min=duration_min,
                duration_max=duration_max,
            )
            bm25_score = self._bm25_score(bm25_terms, idx)
            score = heuristic_score + (0.8 * bm25_score)
            if score <= 0:
                continue

            url = str(item.get("url", "")).strip()
            if url:
                item_id = "assess_" + hashlib.md5(url.encode()).hexdigest()[:12]
            else:
                fallback = f"{item.get('name', '')}:{item.get('test_type', '')}"
                item_id = "assess_kw_" + hashlib.md5(fallback.encode()).hexdigest()[:12]

            scored.append(
                {
                    "id": item_id,
                    "score": float(score),  # raw keyword score
                    "metadata": {
                        "name": item.get("name", ""),
                        "url": item.get("url", ""),
                        "test_type": item.get("test_type", ""),
                        "all_test_types": item.get("all_test_types", ""),
                        "description": item.get("description", ""),
                        "duration": item.get("duration", 0),
                        "remote_support": item.get("remote_support", ""),
                        "adaptive_support": item.get("adaptive_support", ""),
                        "category": item.get("category", ""),
                    },
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _result_identity(self, result: Dict[str, Any]) -> str:
        metadata = result.get("metadata", {}) or {}
        return str(metadata.get("url") or result.get("id") or metadata.get("name") or id(result))

    def _hybrid_fuse(self, semantic_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Fuse semantic and keyword rankings via weighted reciprocal rank fusion.
        """
        if not semantic_results and not keyword_results:
            return []

        # Weighted RRF constants
        rrf_k = 60.0
        semantic_weight = 1.0
        keyword_weight = 0.6

        fused: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(semantic_results, start=1):
            key = self._result_identity(result)
            if key not in fused:
                fused[key] = {"result": result, "score": 0.0}
            fused[key]["score"] += semantic_weight / (rrf_k + rank)

        for rank, result in enumerate(keyword_results, start=1):
            key = self._result_identity(result)
            if key not in fused:
                fused[key] = {"result": result, "score": 0.0}
            fused[key]["score"] += keyword_weight / (rrf_k + rank)
            # Prefer richer semantic metadata when present; otherwise keep keyword item.
            if key in fused and not fused[key]["result"].get("metadata", {}).get("description"):
                fused[key]["result"] = result

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        # Write fused score into `score` field for downstream sort/logic.
        final = []
        for item in ranked:
            result_obj = dict(item["result"])
            result_obj["score"] = float(item["score"])
            final.append(result_obj)
        return final
    
    def retrieve_balanced(
        self,
        query: str,
        top_k: int = 10,
        hard_skill_ratio: float = 0.6,
        min_score: float = -1.0
    ) -> List[Dict]:
        """
        Retrieve balanced mix of hard skills (K) and soft skills (P) assessments
        
        Args:
            query: Search query
            top_k: Total number of results to return
            hard_skill_ratio: Ratio of hard skills (K) to soft skills (P)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of balanced assessment results
        """
        # Relevance-first: take top_k by score to maximize recall, then optionally swap for balance.
        candidate_pool = self.retrieve(
            query=query,
            top_k=max(top_k * 10, 100),
            test_type_filter=None,
            min_score=min_score
        )
        if not candidate_pool:
            return []

        # Start with top_k by relevance (candidate_pool is already sorted by score).
        selected = list(candidate_pool[:top_k])
        selected_ids = {self._result_key(r) for r in selected}
        rest = [r for r in candidate_pool[top_k:] if self._result_key(r) not in selected_ids]

        current_k = sum(1 for r in selected if 'K' in self._extract_type_codes(r))
        current_p = sum(1 for r in selected if 'P' in self._extract_type_codes(r))
        target_k, target_p = self._balance_targets(query, top_k=top_k, hard_skill_ratio=hard_skill_ratio)

        # Optional swaps: if we have 0 K or 0 P, swap the lowest-ranked item for the best missing type.
        def best_of_type(ty: str) -> Optional[Dict]:
            for r in rest:
                if ty in self._extract_type_codes(r):
                    return r
            return None

        if current_k == 0 and target_k >= 1:
            replacement = best_of_type('K')
            if replacement and len(selected) >= 1:
                selected[-1] = replacement
                current_k = 1
                selected_ids.add(self._result_key(replacement))
                rest = [r for r in rest if self._result_key(r) != self._result_key(replacement)]
        if current_p == 0 and target_p >= 1:
            replacement = best_of_type('P')
            if replacement and len(selected) >= 1:
                # Use -2 so we don't overwrite the slot we may have used for K.
                idx = -2 if current_k > 0 and len(selected) >= 2 else -1
                selected[idx] = replacement
                current_p = 1

        # Keep relevance order (no re-sort so top slots stay by score).
        balanced_results = selected[:top_k]
        k_in_results = sum(1 for r in balanced_results if 'K' in self._extract_type_codes(r))
        p_in_results = sum(1 for r in balanced_results if 'P' in self._extract_type_codes(r))
        logger.info(
            "Retrieved balanced set (relevance-first): %s K-tagged, %s P-tagged (targets: %s/%s)",
            k_in_results, p_in_results, target_k, target_p
        )
        return balanced_results

    def _extract_type_codes(self, result: Dict) -> set:
        """Extract normalized type codes from metadata/all_test_types/test_type."""
        metadata = result.get('metadata', {}) or {}
        raw = metadata.get('all_test_types') or metadata.get('test_type') or ''
        tokens = [token for token in re.split(r"[,\s/|]+", str(raw).upper()) if token]
        return {token for token in tokens if re.fullmatch(r'[ABCDEKPS]', token)}

    def _result_key(self, result: Dict) -> str:
        """Return a stable dedupe key for a retrieval result."""
        metadata = result.get('metadata', {}) or {}
        return str(
            result.get('id')
            or metadata.get('url')
            or metadata.get('name')
            or id(result)
        )

    def _balance_targets(self, query: str, top_k: int, hard_skill_ratio: float) -> tuple[int, int]:
        """
        Compute desired K/P target counts.

        For mixed-domain queries (both K and P signals present), use 50/50 split.
        For strongly single-domain queries, bias toward that domain.
        """
        scores = self.preprocessor.domain_scores(query)
        k_score, p_score = scores['K'], scores['P']

        if k_score > 0 and p_score > 0:
            total = k_score + p_score
            k_ratio = (k_score / total) if total else hard_skill_ratio
            k_ratio = max(0.25, min(0.75, k_ratio))
        elif k_score > 0:
            k_ratio = max(hard_skill_ratio, 0.7)
        elif p_score > 0:
            k_ratio = min(hard_skill_ratio, 0.3)
        else:
            k_ratio = hard_skill_ratio

        k_target = int(round(top_k * k_ratio))
        k_target = max(1, min(top_k - 1, k_target)) if top_k > 1 else top_k
        p_target = max(0, top_k - k_target)
        return k_target, p_target
    
    def format_results(self, results: List[Dict]) -> List[Dict]:
        """
        Format retrieval results for API/display
        
        Args:
            results: Raw retrieval results
            
        Returns:
            Formatted results with assessment details
        """
        formatted = []
        for result in results:
            metadata = result.get('metadata', {})
            formatted.append({
                'assessment_name': metadata.get('name', 'Unknown'),
                'assessment_url': metadata.get('url', ''),
                'test_type': metadata.get('test_type', ''),
                'all_test_types': metadata.get('all_test_types', ''),
                'description': metadata.get('description', ''),
                'duration': metadata.get('duration', 0),
                'remote_support': metadata.get('remote_support', ''),
                'adaptive_support': metadata.get('adaptive_support', ''),
                'category': metadata.get('category', ''),
                'from_labeled_prior': metadata.get('from_labeled_prior', ''),
                'similarity_score': result.get('score', 0.0)
            })
        
        return formatted


def create_retriever(vector_db: VectorDB, embedding_model: Optional[str] = None) -> AssessmentRetriever:
    """
    Create AssessmentRetriever instance
    
    Args:
        vector_db: VectorDB instance
        embedding_model: Optional embedding model name
        
    Returns:
        AssessmentRetriever instance
    """
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    return AssessmentRetriever(vector_db, embedding_generator)


if __name__ == "__main__":
    print("Retriever module loaded successfully")
    print("Note: Requires initialized VectorDB and EmbeddingGenerator")
