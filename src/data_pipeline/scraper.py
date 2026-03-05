"""Web scraper for SHL Individual Test Solutions catalog (type=1)."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLScraper:
    """Requests-first scraper for SHL Assessment catalog."""

    def __init__(
        self,
        use_selenium: bool = True,
        headless: bool = True,
        session_rotation_interval: int = 5,
    ):
        """Initialize scraper.

        Notes:
        - `use_selenium`, `headless`, and `session_rotation_interval` are kept for
          backward compatibility with existing CLI arguments.
        - Scraping now runs fully with requests + BeautifulSoup.
        """
        self.base_url = "https://www.shl.com/products/product-catalog/"
        self.catalog_type = "1"  # Individual Test Solutions
        self.page_size = 12
        self.max_pages = 80
        self.use_selenium = use_selenium
        self.headless = headless
        self.session_rotation_interval = session_rotation_interval
        self.timeout = 25

        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Referer": "https://www.shl.com/",
            }
        )

        if use_selenium:
            logger.info("Selenium mode requested but disabled: scraper now uses requests-only implementation")

    def _catalog_page_url(self, start: int) -> str:
        params = {"type": self.catalog_type}
        if start > 0:
            params["start"] = str(start)
        return f"{self.base_url}?{urlencode(params)}"

    def _is_blocked_or_error_page(self, html: str) -> bool:
        text = html.lower()
        if not text:
            return True

        # Positive signatures for the real catalog page; if present, don't treat as blocked.
        if (
            "talent assessments catalog" in text
            or "product-catalogue__table" in text
            or "/products/product-catalog/view/" in text
        ):
            return False

        # Strong signatures from SHL error/blocked templates.
        block_markers = [
            "<title>server error",
            "ss-errorpage",
            "request blocked",
            "we'll try to fix this soon",
            "error 403",
            "access denied",
            "cf-chl-bypass",
        ]
        return any(marker in text for marker in block_markers)

    def _fetch_html(self, url: str, *, max_retries: int = 5) -> Optional[str]:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                status = response.status_code

                if status in {403, 429, 500, 502, 503, 504}:
                    logger.warning("Fetch %s returned status %s (attempt %s/%s)", url, status, attempt, max_retries)
                    self._backoff(attempt)
                    self._rotate_user_agent()
                    continue

                if status >= 400:
                    logger.warning("Fetch %s failed with status %s", url, status)
                    return None

                html = response.text or ""
                if len(html) < 800 or self._is_blocked_or_error_page(html):
                    logger.warning("Fetch %s returned blocked/error content (attempt %s/%s)", url, attempt, max_retries)
                    self._backoff(attempt)
                    self._rotate_user_agent()
                    continue

                return html
            except requests.RequestException as exc:
                logger.warning("Fetch %s failed on attempt %s/%s: %s", url, attempt, max_retries, exc)
                self._backoff(attempt)
                self._rotate_user_agent()

        logger.error("Giving up fetch after %s attempts: %s", max_retries, url)
        return None

    def _backoff(self, attempt: int) -> None:
        # Short exponential backoff with jitter.
        base = min(2**attempt, 12)
        delay = base + random.uniform(0.2, 1.5)
        time.sleep(delay)

    def _rotate_user_agent(self) -> None:
        self.session.headers["User-Agent"] = random.choice(self.user_agents)

    def _find_individual_tests_table(self, soup: BeautifulSoup):
        """Find the Individual Test Solutions table."""
        tables = soup.find_all("table")
        if not tables:
            return None

        for table in tables:
            headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
            joined = " ".join(headers)
            if "individual test solutions" in joined:
                return table
            if "assessment" in joined and "test type" in joined:
                return table

        for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = heading.get_text(" ", strip=True).lower()
            if "individual test" in text:
                table = heading.find_next("table")
                if table is not None:
                    return table

        # Last fallback: use first table found on the page.
        return tables[0]

    def _parse_support_cell(self, cell, default: Optional[str] = None) -> Optional[str]:
        if not cell:
            return default

        class_tokens = set()
        for el in [cell] + cell.find_all(True):
            for cls in (el.get("class") or []):
                class_tokens.add(str(cls).strip().lower())

        if "-yes" in class_tokens:
            return "Yes"
        if "-no" in class_tokens:
            return "No"

        text = cell.get_text(" ", strip=True).lower()
        if re.search(r"\byes\b|supported|available|true|1", text):
            return "Yes"
        if re.search(r"\bno\b|not\s+supported|false|0", text):
            return "No"
        return default

    def _extract_test_types(self, cell) -> Tuple[Optional[str], Optional[str]]:
        key_spans = cell.select(".product-catalogue__key") if cell else []
        if key_spans:
            all_codes = [s.get_text(strip=True).upper() for s in key_spans if s.get_text(strip=True)]
        else:
            text = cell.get_text(" ", strip=True) if cell else ""
            all_codes = re.findall(r"\b[ABCDEKPS]\b", text.upper())

        normalized = [code for code in all_codes if re.fullmatch(r"[ABCDEKPS]", code)]
        if not normalized:
            return None, None

        # Prioritize major families for primary type.
        if "K" in normalized:
            primary = "K"
        elif "P" in normalized:
            primary = "P"
        else:
            primary = normalized[0]

        return primary, " ".join(normalized)

    def _parse_table_row(self, row) -> Optional[Dict]:
        cells = row.find_all("td")
        if len(cells) < 2:
            return None

        link = cells[0].find("a", href=True)
        if link is None:
            return None

        name = link.get_text(" ", strip=True)
        if not name:
            return None

        if "pre-packaged" in name.lower() or "prepackaged" in name.lower():
            return None

        url = link.get("href", "").strip()
        if url and not url.startswith("http"):
            url = f"https://www.shl.com{url}"

        remote_support = self._parse_support_cell(cells[1], default="No") if len(cells) >= 3 else None
        adaptive_support = self._parse_support_cell(cells[2], default="No") if len(cells) >= 4 else None
        test_type, all_test_types = self._extract_test_types(cells[-1])

        return {
            "name": name,
            "url": url,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "test_type": test_type,
            "all_test_types": all_test_types,
        }

    def _parse_catalog_page(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        table = self._find_individual_tests_table(soup)
        if table is None:
            return []

        items: List[Dict] = []
        for row in table.find_all("tr"):
            if row.find("th"):
                continue
            parsed = self._parse_table_row(row)
            if parsed:
                items.append(parsed)

        return items

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            text = " ".join(meta.get("content", "").split())
            if text:
                return self._clean_description(text)

        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            text = " ".join(og.get("content", "").split())
            if text:
                return self._clean_description(text)

        selectors = [
            ".product-summary",
            ".product-description",
            ".field--name-body",
            ".wysiwyg",
            "article p",
            ".typ p",
        ]
        for selector in selectors:
            node = soup.select_one(selector)
            if node:
                text = " ".join(node.get_text(" ", strip=True).split())
                if len(text) >= 30:
                    return self._clean_description(text)

        return None

    def _clean_description(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()

        # Remove trailing catalog metadata that often appears in scraped body text.
        cut_markers = [
            r"\bjob levels\b",
            r"\blanguages\b",
            r"\bassessment length\b",
            r"\btest type\b",
            r"\bremote testing\b",
            r"\badaptive\/irt\b",
            r"\bdownloads\b",
        ]
        lower = text.lower()
        cut_index = len(text)
        for marker in cut_markers:
            match = re.search(marker, lower)
            if match:
                cut_index = min(cut_index, match.start())

        text = text[:cut_index].strip(" ,;:-")
        return text

    def _extract_duration(self, soup: BeautifulSoup) -> Optional[int]:
        text = soup.get_text(" ", strip=True)
        if not text:
            return None

        patterns = [
            r"approximate\s*completion\s*time\s*in\s*minutes\s*[=:]?\s*(\d{1,3})",
            r"(?:assessment\s*length|duration|completion\s*time)\s*[=:]?\s*(\d{1,3})\s*(?:mins?|minutes?)",
            r"\b(\d{1,3})\s*(?:mins?|minutes?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 240:
                return value

        return None

    def _extract_yes_no_from_text(self, text: str, label_pattern: str) -> Optional[str]:
        # Example matches:
        #   Remote Testing: Yes
        #   Adaptive/IRT = No
        regex = re.compile(label_pattern + r"\s*[:=]?\s*(yes|no)", flags=re.IGNORECASE)
        match = regex.search(text)
        if not match:
            return None
        return "Yes" if match.group(1).lower() == "yes" else "No"

    def _extract_yes_no(self, soup: BeautifulSoup, label_keywords: Iterable[str]) -> Optional[str]:
        text = soup.get_text(" ", strip=True)
        if not text:
            return None

        label = "(?:" + "|".join(re.escape(keyword) for keyword in label_keywords) + ")"
        return self._extract_yes_no_from_text(text, label)

    def _fetch_detail_soup(self, url: str) -> Optional[BeautifulSoup]:
        html = self._fetch_html(url, max_retries=4)
        if html is None:
            return None
        return BeautifulSoup(html, "html.parser")

    def _enrich_assessment_details(self, assessments: List[Dict]) -> List[Dict]:
        enriched: List[Dict] = []
        total = len(assessments)

        for idx, assessment in enumerate(assessments, start=1):
            item = dict(assessment)
            url = item.get("url")
            if not url:
                enriched.append(item)
                continue

            soup = self._fetch_detail_soup(url)
            if soup is not None:
                item["description"] = self._extract_description(soup)
                item["duration"] = self._extract_duration(soup)

                if item.get("remote_support") is None:
                    item["remote_support"] = self._extract_yes_no(
                        soup, ["remote testing", "remote support", "remote"]
                    )
                if item.get("adaptive_support") is None:
                    item["adaptive_support"] = self._extract_yes_no(
                        soup, ["adaptive/irt", "adaptive testing", "adaptive support", "adaptive"]
                    )
            else:
                item.setdefault("description", None)
                item.setdefault("duration", None)

            enriched.append(item)
            if idx % 25 == 0 or idx == total:
                logger.info("Detail enrichment progress: %s/%s", idx, total)
            time.sleep(random.uniform(0.15, 0.5))

        return enriched

    def scrape_catalog(self) -> List[Dict]:
        """Scrape catalog pages and enrich records from detail pages."""
        logger.info("Starting SHL scrape from %s?type=%s", self.base_url, self.catalog_type)

        all_items: List[Dict] = []
        seen_urls = set()
        empty_pages = 0

        for page_index in range(self.max_pages):
            start = page_index * self.page_size
            page_url = self._catalog_page_url(start)
            logger.info("Scraping page %s (start=%s)", page_index + 1, start)

            html = self._fetch_html(page_url)
            if html is None:
                logger.warning("Skipping page due to repeated fetch failures: %s", page_url)
                empty_pages += 1
                if empty_pages >= 3:
                    logger.error("Stopping scrape after repeated fetch failures")
                    break
                continue

            page_items = self._parse_catalog_page(html)
            if not page_items:
                empty_pages += 1
                logger.info("No assessment rows found on page %s", page_index + 1)
                if empty_pages >= 2:
                    logger.info("Stopping pagination after %s consecutive empty pages", empty_pages)
                    break
                continue

            empty_pages = 0
            new_count = 0
            for item in page_items:
                url = item.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                all_items.append(item)
                new_count += 1

            logger.info("Page %s yielded %s new assessments", page_index + 1, new_count)

            # A page with no new URLs means we've reached the end.
            if new_count == 0:
                logger.info("No new URLs found; stopping pagination")
                break

            time.sleep(random.uniform(0.8, 1.8))

        logger.info("Collected %s unique assessments before detail enrichment", len(all_items))
        enriched = self._enrich_assessment_details(all_items)
        logger.info("Scrape complete: %s assessments", len(enriched))
        return enriched

    def close(self):
        """No-op kept for API compatibility with previous Selenium implementation."""
        return


def scrape_shl_catalog(
    output_path: str = "data/raw_catalog.json",
    use_selenium: bool = True,
    session_rotation_interval: int = 5,
    headless: bool = True,
) -> List[Dict]:
    """Scrape SHL catalog and persist raw JSON output."""
    scraper = SHLScraper(
        use_selenium=use_selenium,
        headless=headless,
        session_rotation_interval=session_rotation_interval,
    )

    try:
        assessments = scraper.scrape_catalog()
        output_data = {
            "scraped_at": datetime.now().isoformat(),
            "source_url": f"{scraper.base_url}?type={scraper.catalog_type}",
            "total_count": len(assessments),
            "assessments": assessments,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved %s assessments to %s", len(assessments), output_path)
        return assessments
    finally:
        scraper.close()


if __name__ == "__main__":
    results = scrape_shl_catalog()
    print(f"Scraped {len(results)} Individual Test Solutions")
