"""
Selenium-based Stocktwits scraper - Most reliable method
Requires: pip install selenium webdriver-manager
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import re
import random

from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.config import Config

class SeleniumStocktwitsScraper:
    """Use Selenium to scrape Stocktwits - bypasses most blocks"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.nlp_analyzer = None
        self.api_responses = []  # Store intercepted API responses
        
        if Config.USE_NLP_SENTIMENT:
            try:
                self.nlp_analyzer = SentimentAnalyzer(method=Config.NLP_SENTIMENT_METHOD)
            except:
                pass
    
    def _init_driver(self):
        """Initialize Chrome driver with anti-detection measures"""
        if self.driver:
            return
        
        chrome_options = Options()
        
        # Anti-detection options
        if self.headless:
            chrome_options.add_argument('--headless=new')  # Use new headless mode
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Realistic user agent
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Enable performance logging for network interception
        chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
        
        # Window size to avoid detection
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Disable automation flags
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set timeouts
            self.driver.set_page_load_timeout(30)  # 30 seconds for page load
            self.driver.implicitly_wait(10)  # 10 seconds for element finding
            
            # Execute anti-detection scripts
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    window.chrome = {
                        runtime: {}
                    };
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                '''
            })
            
            print("  Chrome driver initialized successfully")
        except Exception as e:
            print(f"Error initializing Selenium: {e}")
            print("Install: pip install selenium webdriver-manager")
            raise
    
    def _intercept_network_requests(self):
        """Intercept network requests to capture API responses"""
        try:
            # Enable network domain
            self.driver.execute_cdp_cmd('Network.enable', {})
            
            # Store intercepted responses
            self.api_responses = []
        except Exception as e:
            print(f"  Warning: Could not enable network interception: {e}")
    
    def _get_api_responses_from_logs(self) -> List[Dict]:
        """Extract API responses from performance logs"""
        api_responses = []
        try:
            logs = self.driver.get_log('performance')
            for log in logs:
                try:
                    log_message = json.loads(log['message'])
                    method = log_message.get('message', {}).get('method', '')
                    
                    # Look for network responses
                    if method == 'Network.responseReceived':
                        response = log_message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        # Check if it's a Stocktwits API endpoint
                        if 'api.stocktwits.com' in url or 'stocktwits.com/api' in url:
                            # Try to get response body
                            request_id = log_message.get('message', {}).get('params', {}).get('requestId', '')
                            if request_id:
                                try:
                                    response_body = self.driver.execute_cdp_cmd('Network.getResponseBody', {
                                        'requestId': request_id
                                    })
                                    if response_body:
                                        api_responses.append({
                                            'url': url,
                                            'body': response_body.get('body', '')
                                        })
                                except:
                                    pass
                except Exception as e:
                    continue
        except Exception as e:
            print(f"  Warning: Could not read performance logs: {e}")
        
        return api_responses
    
    def get_messages(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get messages using Selenium with multiple extraction methods"""
        if not self.driver:
            self._init_driver()
        
        messages = []
        url = f"https://stocktwits.com/symbol/{symbol}"
        
        try:
            print(f"  Loading page: {url}")
            
            # Enable network interception before loading page
            try:
                self._intercept_network_requests()
            except Exception as e:
                print(f"  Warning: Could not intercept network: {e}")
            
            # Load page with timeout handling
            page_loaded = False
            try:
                print(f"  Loading page (timeout: 30s)...")
                self.driver.get(url)
                print(f"  Page loaded successfully")
                page_loaded = True
            except TimeoutException:
                print(f"  Warning: Page load timed out, checking if partially loaded...")
                # Check if page partially loaded
                try:
                    current_url = self.driver.current_url
                    page_source_length = len(self.driver.page_source)
                    if page_source_length > 1000:  # Some content loaded
                        print(f"  Page partially loaded ({page_source_length} chars), continuing...")
                        page_loaded = True
                    else:
                        print(f"  Page not loaded, retrying...")
                        raise
                except:
                    print(f"  Page not accessible, trying alternative method...")
                    raise
            
            if not page_loaded:
                raise Exception("Could not load page")
            
            time.sleep(random.uniform(2, 4))  # Random wait
            
            # Scroll to trigger lazy loading (with error handling)
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
                time.sleep(random.uniform(1, 2))
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                print(f"  Warning: Could not scroll: {e}")
                # Continue anyway
            
            # Method 1: Try to extract from intercepted API responses
            print("  Attempting to extract from API responses...")
            api_responses = self._get_api_responses_from_logs()
            
            for response in api_responses:
                try:
                    if response.get('body'):
                        body_text = response['body']
                        # Try to parse as JSON
                        if body_text.startswith('{') or body_text.startswith('['):
                            data = json.loads(body_text)
                            msgs = self._extract_messages_from_dict(data)
                            if msgs:
                                messages.extend(msgs)
                                print(f"  Found {len(msgs)} messages from API response")
                except Exception as e:
                    continue
            
            # Method 2: Extract JSON from page source (embedded data)
            if len(messages) < limit:
                print("  Attempting to extract from page source...")
                page_source = self.driver.page_source
                
                # Look for embedded JSON data
                patterns = [
                    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
                    r'window\.__APOLLO_STATE__\s*=\s*({.*?});',
                    r'window\.__NEXT_DATA__\s*=\s*({.*?});',
                    r'"messages"\s*:\s*(\[.*?\])',
                    r'data-messages=["\'](\[.*?\])["\']',
                ]
                
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, page_source, re.DOTALL)
                        for match in matches:
                            try:
                                json_str = match.group(1) if match.groups() else match.group(0)
                                # Clean up JSON string
                                json_str = json_str.strip().rstrip(';').strip()
                                if json_str.startswith('{') or json_str.startswith('['):
                                    data = json.loads(json_str)
                                    msgs = self._extract_messages_from_dict(data)
                                    if msgs:
                                        messages.extend(msgs)
                                        print(f"  Found {len(msgs)} messages from embedded JSON")
                                        break
                            except json.JSONDecodeError:
                                continue
                    except Exception as e:
                        continue
            
            # Method 3: Parse HTML messages directly (fallback)
            if len(messages) < limit:
                print("  Attempting to extract from HTML elements...")
                try:
                    # Wait for messages to appear
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    
                    # Try multiple selectors for message elements
                    selectors = [
                        '[data-testid*="message"]',
                        '[class*="message"]',
                        '[class*="Message"]',
                        '[class*="stream"]',
                        '[class*="Stream"]',
                        'article',
                        '[role="article"]',
                        '.st_Message',
                        '.message-body',
                    ]
                    
                    message_elements = []
                    for selector in selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                message_elements = elements
                                print(f"  Found {len(elements)} elements with selector: {selector}")
                                break
                        except:
                            continue
                    
                    for elem in message_elements[:limit]:
                        try:
                            # Try to get text content
                            text = elem.text.strip()
                            
                            # Try to get timestamp if available
                            timestamp = datetime.now()
                            try:
                                time_elem = elem.find_element(By.CSS_SELECTOR, '[class*="time"], [class*="date"], time')
                                time_text = time_elem.get_attribute('datetime') or time_elem.text
                                if time_text:
                                    timestamp = pd.to_datetime(time_text)
                            except:
                                pass
                            
                            if text and len(text) > 10:
                                # Skip if it looks like UI element
                                if any(skip in text.lower() for skip in ['follow', 'like', 'share', 'comment', 'more']):
                                    if len(text) < 50:  # Likely UI element
                                        continue
                                
                                messages.append({
                                    'body': text,
                                    'timestamp': timestamp,
                                    'id': hash(text) % 10000000,
                                    'sentiment': self._extract_sentiment_from_text(text),
                                    'user_id': 0
                                })
                        except Exception as e:
                            continue
                    
                    if len(messages) > 0:
                        print(f"  Found {len(messages)} messages from HTML parsing")
                except TimeoutException:
                    print("  Timeout waiting for page elements")
                except Exception as e:
                    print(f"  Error parsing HTML: {e}")
            
            # Remove duplicates based on body text
            seen_bodies = set()
            unique_messages = []
            for msg in messages:
                body_hash = hash(msg['body'][:100])  # Use first 100 chars for deduplication
                if body_hash not in seen_bodies:
                    seen_bodies.add(body_hash)
                    unique_messages.append(msg)
            
            messages = unique_messages[:limit]
            
            print(f"  Total unique messages extracted: {len(messages)}")
        
        except WebDriverException as e:
            print(f"  WebDriver error: {e}")
        except Exception as e:
            print(f"  Selenium error: {e}")
            import traceback
            traceback.print_exc()
        
        return messages
    
    def get_historical_messages(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get historical messages by scrolling and loading more content
        """
        if not self.driver:
            self._init_driver()
        
        all_messages = []
        url = f"https://stocktwits.com/symbol/{symbol}"
        
        try:
            print(f"  Loading page: {url}")
            self._intercept_network_requests()
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))
            
            # Scroll to load more messages
            scroll_pause_time = random.uniform(1, 2)
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scrolls = 10  # Limit scrolling to avoid infinite loop
            
            while scroll_attempts < max_scrolls:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                # Check if new content loaded
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
            
            # Extract messages after scrolling
            messages = self.get_messages(symbol, limit=200)  # Get more messages
            all_messages.extend(messages)
            
        except Exception as e:
            print(f"  Error getting historical messages: {e}")
        
        if len(all_messages) > 0:
            df = pd.DataFrame(all_messages)
            # Filter by date if needed
            if 'timestamp' in df.columns:
                cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
                df = df[df['timestamp'] >= cutoff_date]
            return df
        
        return pd.DataFrame()
    
    def _extract_messages_from_dict(self, data: dict, depth: int = 0) -> List[Dict]:
        """Recursively find messages in nested dictionaries"""
        if depth > 10:  # Increased depth limit
            return []
        
        messages = []
        
        if isinstance(data, dict):
            # Check for messages array directly
            if 'messages' in data and isinstance(data['messages'], list):
                for msg in data['messages']:
                    if isinstance(msg, dict):
                        extracted = self._extract_single_message(msg)
                        if extracted:
                            messages.append(extracted)
            
            # Check for data.messages pattern
            if 'data' in data and isinstance(data['data'], dict):
                if 'messages' in data['data']:
                    for msg in data['data']['messages']:
                        if isinstance(msg, dict):
                            extracted = self._extract_single_message(msg)
                            if extracted:
                                messages.append(extracted)
            
            # Check for symbol.messages pattern
            if 'symbol' in data and isinstance(data['symbol'], dict):
                if 'messages' in data['symbol']:
                    for msg in data['symbol']['messages']:
                        if isinstance(msg, dict):
                            extracted = self._extract_single_message(msg)
                            if extracted:
                                messages.append(extracted)
            
            # Recursively search nested structures
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    result = self._extract_messages_from_dict(value if isinstance(value, dict) else {'messages': value}, depth + 1)
                    if result:
                        messages.extend(result)
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    result = self._extract_messages_from_dict(item, depth + 1)
                    if result:
                        messages.extend(result)
        
        return messages
    
    def _extract_single_message(self, msg: Dict) -> Optional[Dict]:
        """Extract a single message from message dict"""
        try:
            # Try different field names for body
            body = msg.get('body') or msg.get('text') or msg.get('message') or msg.get('content')
            if not body or len(str(body).strip()) < 5:
                return None
            
            # Parse timestamp
            timestamp = datetime.now()
            timestamp_fields = ['created_at', 'timestamp', 'date', 'time', 'createdAt']
            for field in timestamp_fields:
                if field in msg and msg[field]:
                    try:
                        timestamp_str = str(msg[field])
                        # Try different formats
                        for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt)
                                break
                            except:
                                continue
                        timestamp = pd.to_datetime(timestamp_str)
                        break
                    except:
                        continue
            
            # Extract sentiment
            sentiment = self._extract_sentiment_from_msg(msg)
            
            # Get ID
            msg_id = msg.get('id') or msg.get('message_id') or hash(str(body)) % 10000000
            
            # Get user ID
            user_id = 0
            if 'user' in msg and isinstance(msg['user'], dict):
                user_id = msg['user'].get('id', 0)
            elif 'user_id' in msg:
                user_id = msg['user_id']
            
            return {
                'body': str(body).strip(),
                'timestamp': timestamp,
                'id': int(msg_id),
                'sentiment': sentiment,
                'user_id': int(user_id)
            }
        except Exception as e:
            return None
    
    def _extract_sentiment_from_msg(self, msg: Dict) -> float:
        """Extract sentiment from message dict"""
        base = 0.0
        
        # Try to get sentiment from entities
        if 'entities' in msg and isinstance(msg['entities'], dict):
            sentiment_dict = msg['entities'].get('sentiment', {})
            if isinstance(sentiment_dict, dict):
                basic = sentiment_dict.get('basic', '')
                if basic == 'bullish':
                    base = 1.0
                elif basic == 'bearish':
                    base = -1.0
        
        # Try direct sentiment field
        if base == 0.0:
            sentiment_val = msg.get('sentiment') or msg.get('sentiment_score')
            if sentiment_val:
                try:
                    base = float(sentiment_val)
                    base = max(-1.0, min(1.0, base))  # Clamp to [-1, 1]
                except:
                    pass
        
        # Get body text for NLP analysis
        body = msg.get('body') or msg.get('text') or msg.get('message') or msg.get('content') or ''
        
        # Add NLP sentiment if available
        if self.nlp_analyzer and body:
            try:
                nlp_sent = self.nlp_analyzer.analyze(str(body))
                # Weighted combination: 60% Stocktwits, 40% NLP
                combined = 0.6 * base + 0.4 * nlp_sent
                return max(-1.0, min(1.0, combined))  # Clamp to [-1, 1]
            except:
                return base
        
        return base
    
    def _extract_sentiment_from_text(self, text: str) -> float:
        """Extract sentiment from text"""
        if self.nlp_analyzer:
            try:
                return self.nlp_analyzer.analyze(text)
            except:
                return 0.0
        return 0.0
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None

