"""
Web Scraper for SHL Product Catalog
Extracts Individual Test Solutions from https://www.shl.com/products/product-catalog/?type=1

CRITICAL: Uses type=1 URL parameter to get ONLY Individual Test Solutions
- type=1 = Individual Test Solutions ✓
- type=2 = Pre-packaged Job Solutions ✗

Best Practices Implementation:
- Next button clicks (not direct URL navigation)
- Session rotation (restart browser every N pages)
- Human-like behavior (scrolling, random delays)
- Stealth mode to avoid detection
- Debug HTML saving for failed pages
"""

import time
import random
import json
import logging
import re
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHLScraper:
    """Advanced Scraper for SHL Assessment Catalog with Anti-Bot Evasion"""
    
    def __init__(self, use_selenium: bool = True, headless: bool = True, session_rotation_interval: int = 5):
        """
        Initialize the scraper
        
        Args:
            use_selenium: Whether to use Selenium for dynamic content
            headless: Run browser in headless mode
            session_rotation_interval: Restart browser every N pages (default: 5)
        """
        self.base_url = "https://www.shl.com/products/product-catalog/?type=1"  # type=1 for Individual Test Solutions
        self.use_selenium = use_selenium
        self.driver = None
        self.headless = headless
        self.session_rotation_interval = session_rotation_interval
        self.current_page = 1
        self.debug_dir = Path("data/debug_html")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        if use_selenium:
            self._setup_selenium(headless)
    
    def _setup_selenium(self, headless: bool = True):
        """Setup Selenium WebDriver with stealth options"""
        try:
            chrome_options = Options()
            
            # Stealth options to avoid detection
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Realistic user agent
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Window size (even in headless)
            chrome_options.add_argument('--window-size=1920,1080')
            
            if headless:
                chrome_options.add_argument('--headless=new')  # New headless mode
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute stealth script to hide webdriver property
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                '''
            })
            
            logger.info("Selenium WebDriver initialized with stealth mode")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            raise
    
    def _restart_browser(self):
        """Restart browser to get fresh session"""
        logger.info("Restarting browser for fresh session...")
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        time.sleep(random.uniform(2, 4))
        self._setup_selenium(self.headless)
        logger.info("Browser restarted successfully")
    
    def _human_like_delay(self, min_seconds: float = 2.0, max_seconds: float = 5.0):
        """Random delay to simulate human behavior"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def _human_like_scroll(self):
        """Simulate human-like scrolling behavior"""
        if not self.driver:
            return
        
        try:
            # Scroll down gradually
            scroll_pause = random.uniform(0.5, 1.5)
            for i in range(random.randint(2, 4)):
                scroll_amount = random.randint(300, 600)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(scroll_pause)
            
            # Sometimes scroll back up a bit
            if random.random() < 0.3:
                self.driver.execute_script("window.scrollBy(0, -200);")
                time.sleep(random.uniform(0.3, 0.8))
        except Exception as e:
            logger.debug(f"Error during scrolling: {e}")
    
    def _dismiss_cookie_banner(self):
        """Dismiss cookie consent banner if present"""
        if not self.use_selenium or not self.driver:
            return
        
        try:
            self._human_like_delay(1.5, 3.0)
            
            strategies = [
                lambda: self.driver.find_element(By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"),
                lambda: self.driver.find_element(By.XPATH, 
                    "//button[contains(text(), 'Accept') or contains(text(), 'I understand') or contains(text(), 'Continue')]"),
                lambda: self.driver.find_element(By.CSS_SELECTOR, 
                    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll, .cookie-accept, .accept-cookies"),
            ]
            
            for strategy in strategies:
                try:
                    button = strategy()
                    if button.is_displayed():
                        # Scroll to button first
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        self._human_like_delay(0.5, 1.0)
                        button.click()
                        self._human_like_delay(1.0, 2.0)
                        logger.debug("Cookie banner dismissed")
                        return
                except NoSuchElementException:
                    continue
            
            # Fallback: Hide with JavaScript
            try:
                self.driver.execute_script("""
                    var banner = document.getElementById('CybotCookiebotDialogHeader');
                    if (banner) banner.style.display = 'none';
                    var overlay = document.getElementById('CybotCookiebotDialog');
                    if (overlay) overlay.style.display = 'none';
                """)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Could not dismiss cookie banner: {e}")
    
    def _load_initial_page(self) -> bool:
        """Load the initial catalog page"""
        try:
            logger.info(f"Loading initial page: {self.base_url}")
            self.driver.get(self.base_url)
            self._human_like_delay(3, 5)
            
            # Dismiss cookie banner
            self._dismiss_cookie_banner()
            
            # Human-like scroll
            self._human_like_scroll()
            
            # Simulate reading the page
            self._human_like_delay(2.0, 4.0)
            
            # Wait for table
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except TimeoutException:
                logger.warning("Table not found within timeout")
            
            return True
        except Exception as e:
            logger.error(f"Error loading initial page: {e}")
            return False
    
    def _click_next_button(self) -> bool:
        """Click the Next pagination button (human-like)"""
        if not self.driver:
            return False
        
        try:
            # Find Next button with multiple strategies
            next_button = None
            
            strategies = [
                lambda: self.driver.find_element(By.LINK_TEXT, "Next"),
                lambda: self.driver.find_element(By.PARTIAL_LINK_TEXT, "Next"),
                lambda: self.driver.find_element(By.XPATH, "//a[contains(@class, 'pagination') and contains(text(), 'Next')]"),
                lambda: self.driver.find_element(By.XPATH, "//a[@aria-label='Next' or @aria-label='next']"),
                lambda: self.driver.find_element(By.XPATH, "//a[contains(@href, 'start') and contains(text(), 'Next')]"),
                lambda: self.driver.find_element(By.XPATH, "//a[contains(@class, 'pagination__link') and contains(text(), 'Next')]"),
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    next_button = strategy()
                    if next_button and next_button.is_displayed() and next_button.is_enabled():
                        logger.debug(f"Found Next button using strategy {i+1}")
                        break
                except (NoSuchElementException, Exception) as e:
                    logger.debug(f"Strategy {i+1} failed: {e}")
                    continue
            
            if not next_button:
                logger.warning("Next button not found with any strategy")
                # Fallback: Try to find by looking at all pagination links
                try:
                    all_pagination_links = self.driver.find_elements(By.CSS_SELECTOR, 
                        'a.pagination__link, .pagination a, .pager a')
                    for link in all_pagination_links:
                        if 'next' in link.text.lower() or 'next' in link.get_attribute('aria-label', '').lower():
                            next_button = link
                            break
                except:
                    pass
            
            if not next_button:
                logger.error("Could not find Next button with any method")
                return False
            
            # Scroll to button
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                self._human_like_delay(1.0, 2.0)
            except Exception as e:
                logger.debug(f"Error scrolling to button: {e}")
            
            # Use ActionChains for more human-like click
            try:
                ActionChains(self.driver).move_to_element(next_button).pause(random.uniform(0.2, 0.5)).click().perform()
            except Exception as e:
                # Fallback to direct click
                logger.debug(f"ActionChains failed, using direct click: {e}")
                next_button.click()
            
            # Wait for page to load
            self._human_like_delay(3, 6)
            
            # Dismiss cookie banner if it reappears
            self._dismiss_cookie_banner()
            
            # Human-like scroll on new page
            self._human_like_scroll()
            
            # Wait for table
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except TimeoutException:
                logger.warning("Table not found after clicking Next")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clicking Next button: {e}")
            return False
    
    def _save_debug_html(self, page_num: int, content: str):
        """Save HTML for debugging failed pages"""
        debug_file = self.debug_dir / f"page_{page_num}.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved debug HTML to {debug_file}")
    
    def _find_individual_tests_table(self, soup: BeautifulSoup) -> Optional:
        """
        Find the Individual Test Solutions table
        
        Since we're using ?type=1 URL parameter, the page should only show
        Individual Test Solutions table. We just need to find the main table.
        """
        tables = soup.find_all('table')
        logger.debug(f"Found {len(tables)} tables on page")
        
        if not tables:
            return None
        
        # Strategy 1: Look for table with "Test Type" header (main catalog table)
        for idx, table in enumerate(tables):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            
            if any('Test Type' in h for h in headers):
                # Double-check this is NOT pre-packaged by checking first few rows
                is_prepackaged = False
                for row in table.find_all('tr')[:5]:
                    row_text = row.get_text().lower()
                    if 'pre-packaged' in row_text or 'prepackaged' in row_text:
                        is_prepackaged = True
                        logger.warning(f"Table {idx} contains pre-packaged items, skipping")
                        break
                
                if not is_prepackaged:
                    logger.debug(f"Found Individual Test Solutions table (table {idx})")
                    return table
        
        # Strategy 2: Look for the "Individual Test Solutions" heading
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            heading_text = heading.get_text(strip=True)
            if 'Individual Test Solutions' in heading_text or 'Individual Tests' in heading_text:
                next_table = heading.find_next('table')
                if next_table:
                    logger.debug(f"Found Individual Test Solutions table by heading: '{heading_text}'")
                    return next_table
        
        # Strategy 3: First table (since we're on type=1 page, it should be the right one)
        if tables:
            logger.debug("Using first table (type=1 page should only have Individual Tests)")
            return tables[0]
        
        logger.error("Could not find any table on page")
        return None
    
    def _parse_table_row(self, row) -> Optional[Dict]:
        """Parse a table row to extract assessment information"""
        try:
            cells = row.find_all('td')
            if len(cells) < 2:
                return None
            
            assessment = {}
            
            # First cell: name and URL
            first_cell = cells[0]
            link = first_cell.find('a', href=True)
            
            if not link:
                return None
            
            assessment['name'] = link.get_text(strip=True)
            assessment['url'] = link.get('href', '')
            
            # CRITICAL: Filter out Pre-packaged Job Solutions
            name_lower = assessment['name'].lower()
            if 'pre-packaged' in name_lower or 'prepackaged' in name_lower or 'job solution' in name_lower:
                logger.debug(f"Skipping Pre-packaged solution: {assessment['name']}")
                return None
            
            # Make URL absolute
            if assessment['url'] and not assessment['url'].startswith('http'):
                assessment['url'] = f"https://www.shl.com{assessment['url']}"
            
            # Extract test type from last column
            test_type_cell = cells[-1]
            test_type_text = test_type_cell.get_text(strip=True)
            test_type_codes = re.findall(r'[ABCDEKPS]', test_type_text)
            
            if test_type_codes:
                if 'K' in test_type_codes:
                    assessment['test_type'] = 'K'
                elif 'P' in test_type_codes:
                    assessment['test_type'] = 'P'
                else:
                    assessment['test_type'] = test_type_codes[0]
                assessment['all_test_types'] = ' '.join(test_type_codes)
            else:
                assessment['test_type'] = None
            
            return assessment
            
        except Exception as e:
            logger.debug(f"Error parsing table row: {e}")
            return None
    
    def _get_pagination_info(self, soup: BeautifulSoup) -> Tuple[int, int]:
        """Extract pagination information - focus on Individual Test Solutions pagination"""
        total_pages = 32  # Default based on website structure
        
        # Strategy 1: Look for pagination in Individual Test Solutions section only
        if self.use_selenium and self.driver:
            try:
                # Find pagination links specifically in the Individual Test Solutions table area
                # Look for pagination after the second table (Individual Test Solutions)
                pagination_containers = self.driver.find_elements(By.CSS_SELECTOR, 
                    '.pagination, .pager, nav[aria-label*="pagination"]')
                
                selenium_nums = []
                for container in pagination_containers:
                    # Get all links in pagination
                    links = container.find_elements(By.TAG_NAME, 'a')
                    for link in links:
                        text = link.text.strip()
                        # Only consider numeric page numbers (not "Previous", "Next", etc.)
                        if text.isdigit() and 1 <= int(text) <= 50:  # Reasonable range
                            selenium_nums.append(int(text))
                
                if selenium_nums:
                    total_pages = max(selenium_nums)
                    logger.info(f"Found pagination: max page = {total_pages}")
            except Exception as e:
                logger.debug(f"Could not get pagination from Selenium: {e}")
        
        # Strategy 2: Look for ellipsis pattern (e.g., "1 2 3 ... 32")
        if self.use_selenium and self.driver:
            try:
                page_text = self.driver.find_element(By.TAG_NAME, 'body').text
                # Look for pattern like "... 32" or "… 32"
                ellipsis_pattern = re.search(r'[……]|\.\.\.\s*(\d+)', page_text)
                if ellipsis_pattern and ellipsis_pattern.group(1):
                    found_pages = int(ellipsis_pattern.group(1))
                    if 20 <= found_pages <= 50:  # Reasonable range for SHL
                        total_pages = found_pages
                        logger.info(f"Found pagination via ellipsis: {total_pages}")
            except:
                pass
        
        # Strategy 3: Check URL parameters in pagination links
        if self.use_selenium and self.driver:
            try:
                all_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="start"]')
                max_start = 0
                for link in all_links:
                    href = link.get_attribute('href')
                    if href:
                        match = re.search(r'start=(\d+)', href)
                        if match:
                            start_val = int(match.group(1))
                            # Calculate page: start=0 is page 1, start=12 is page 2, etc.
                            # So page = (start / 12) + 1
                            page_num = (start_val // 12) + 1
                            if page_num > max_start:
                                max_start = page_num
                
                if max_start > 0 and max_start <= 50:
                    total_pages = max_start
                    logger.info(f"Found pagination via URL params: {total_pages}")
            except:
                pass
        
        logger.info(f"Detected {total_pages} total pages")
        return 1, total_pages
    
    def scrape_catalog(self) -> List[Dict]:
        """Main method to scrape the SHL product catalog using Next button clicks"""
        logger.info(f"Starting to scrape SHL catalog from {self.base_url}")
        assessments = []
        
        try:
            # Load initial page
            if not self._load_initial_page():
                logger.error("Failed to load initial page")
                return assessments
            
            # Get page content and find pagination
            page_content = self.driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')
            current_page, total_pages = self._get_pagination_info(soup)
            
            # Scrape all pages
            consecutive_empty_pages = 0
            max_consecutive_empty = 3  # Stop if 3 consecutive pages are empty
            
            for page_num in range(1, total_pages + 1):
                logger.info(f"Scraping page {page_num}/{total_pages}")
                
                # Rotate session every N pages (but not on first page)
                if page_num > 1 and page_num % self.session_rotation_interval == 1:
                    logger.info(f"Session rotation: restarting browser (every {self.session_rotation_interval} pages)")
                    # Restart browser for fresh session
                    self._restart_browser()
                    # Navigate to the page we need using URL (one-time after restart)
                    # Then continue with Next button clicks
                    # type=1 is for Individual Test Solutions (type=2 is Pre-packaged)
                    resume_url = f"https://www.shl.com/products/product-catalog/?start={(page_num - 1) * 12}&type=1"
                    logger.info(f"Resuming at page {page_num} via URL: {resume_url}")
                    self.driver.get(resume_url)
                    self._human_like_delay(3, 5)
                    self._dismiss_cookie_banner()
                    self._human_like_scroll()
                    
                    # Get page content directly (skip Next button click this time)
                    page_content = self.driver.page_source
                    soup = BeautifulSoup(page_content, 'html.parser')
                    table = self._find_individual_tests_table(soup)
                    
                    if table:
                        rows = table.find_all('tr')
                        page_assessments = []
                        for row in rows:
                            if row.find('th'):
                                continue
                            assessment = self._parse_table_row(row)
                            if assessment:
                                page_assessments.append(assessment)
                        
                        logger.info(f"Found {len(page_assessments)} assessments on page {page_num} (after restart)")
                        if page_assessments:
                            assessments.extend(page_assessments)
                            consecutive_empty_pages = 0
                        else:
                            consecutive_empty_pages += 1
                            self._save_debug_html(page_num, page_content)
                        
                        # Continue to next iteration
                        if page_num < total_pages:
                            delay = random.uniform(2.0, 4.0)
                            time.sleep(delay)
                        continue
                
                # Normal flow: get page content
                
                # Get current page content
                if page_num == 1:
                    page_content = self.driver.page_source
                else:
                    # Click Next button for subsequent pages
                    if not self._click_next_button():
                        logger.error(f"Failed to navigate to page {page_num}")
                        break
                    page_content = self.driver.page_source
                
                # Parse page
                soup = BeautifulSoup(page_content, 'html.parser')
                table = self._find_individual_tests_table(soup)
                
                if not table:
                    logger.warning(f"No table found on page {page_num}")
                    self._save_debug_html(page_num, page_content)
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_consecutive_empty:
                        logger.error(f"Too many consecutive empty pages ({consecutive_empty_pages}), stopping")
                        break
                    continue
                
                # Parse table rows
                rows = table.find_all('tr')
                page_assessments = []
                
                for row in rows:
                    if row.find('th'):
                        continue
                    assessment = self._parse_table_row(row)
                    if assessment:
                        page_assessments.append(assessment)
                
                logger.info(f"Found {len(page_assessments)} assessments on page {page_num}")
                
                if len(page_assessments) == 0:
                    consecutive_empty_pages += 1
                    logger.warning(f"Empty page {page_num} (consecutive: {consecutive_empty_pages})")
                    self._save_debug_html(page_num, page_content)
                    
                    if consecutive_empty_pages >= max_consecutive_empty:
                        logger.error(f"Too many consecutive empty pages ({consecutive_empty_pages}), stopping")
                        break
                else:
                    consecutive_empty_pages = 0  # Reset counter
                    assessments.extend(page_assessments)
                
                # Random delay between pages (longer for later pages)
                if page_num < total_pages:
                    # Progressive delays: start at 3-6s, increase to 5-10s, then 7-15s
                    if page_num < 5:
                        base_delay = 3.0
                        max_multiplier = 2.0
                    elif page_num < 10:
                        base_delay = 5.0
                        max_multiplier = 2.0
                    else:
                        base_delay = 7.0
                        max_multiplier = 2.2
                    
                    delay = random.uniform(base_delay, base_delay * max_multiplier)
                    logger.debug(f"Waiting {delay:.2f}s before next page")
                    time.sleep(delay)
            
            logger.info(f"Scraped {len(assessments)} Individual Test Solutions from {total_pages} pages")
            
            # Remove duplicates
            seen_urls = set()
            unique_assessments = []
            for assessment in assessments:
                url = assessment.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_assessments.append(assessment)
            
            logger.info(f"After deduplication: {len(unique_assessments)} unique assessments")
            return unique_assessments
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
            return assessments
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def close(self):
        """Close browser and cleanup"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


def scrape_shl_catalog(
    output_path: str = "data/raw_catalog.json", 
    use_selenium: bool = True,
    session_rotation_interval: int = 5,
    headless: bool = False
) -> List[Dict]:
    """
    Convenience function to scrape SHL catalog and save results
    
    Args:
        output_path: Path to save scraped data
        use_selenium: Whether to use Selenium
        session_rotation_interval: Restart browser every N pages
        headless: Run browser in headless mode (default: False for better stealth)
        
    Returns:
        List of assessment dictionaries
    """
    scraper = SHLScraper(use_selenium=use_selenium, headless=headless, session_rotation_interval=session_rotation_interval)
    
    try:
        assessments = scraper.scrape_catalog()
        
        # Add metadata
        output_data = {
            'scraped_at': datetime.now().isoformat(),
            'source_url': scraper.base_url,
            'total_count': len(assessments),
            'assessments': assessments
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(assessments)} assessments to {output_path}")
        
        return assessments
        
    finally:
        scraper.close()


if __name__ == "__main__":
    # Run scraper
    assessments = scrape_shl_catalog()
    print(f"\nScraped {len(assessments)} Individual Test Solutions")
