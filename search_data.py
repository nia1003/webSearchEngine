import requests
import json
import re
import io
import os
import tldextract
from bs4 import BeautifulSoup
from PIL import Image
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from duckduckgo_search import DDGS
from tqdm import tqdm
from transformers import pipeline

# ========== Zero-Shot Classification ==========
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def get_relate_domain_score(text, binary_labels):
    """
    取得「Academic Paper」的機率分數 (0~1 之間)。
    若內容為空，直接回傳 0.0。
    """
    if not text.strip():
        return 0.0
    target_labels = binary_labels.split(" ")
    if len(target_labels) >= 2:
        labels = [target_labels[0], target_labels[1]]
    else:
        raise ValueError("target_labels must contain at least two elements.")
    result = classifier(text, labels)
    # 找到 "Academic Paper" 對應的分數
    relation_idx = result["labels"].index(target_labels[0])
    relate_domain_score = result["scores"][relation_idx]
    return relate_domain_score

# ========== (可選) 機器人排除協議解析 ==========
try:
    from robotexclusionrulesparser import RobotExclusionRulesParser
    ROBOTS_PARSER_AVAILABLE = True
except ImportError:
    ROBOTS_PARSER_AVAILABLE = False

def can_crawl(url):
    """
    簡易示範：抓取該網域的 robots.txt，檢查是否允許爬取該路徑。
    """
    if not ROBOTS_PARSER_AVAILABLE:
        return True
    
    ext = tldextract.extract(url)
    domain = ext.registered_domain
    if not domain:
        return True

    robots_url = f"http://{domain}/robots.txt"
    try:
        resp = requests.get(robots_url, timeout=15)
        if resp.status_code == 200:
            rp = RobotExclusionRulesParser()
            rp.parse(resp.text)
            return rp.is_allowed("*", url)
        else:
            return True
    except Exception:
        return True

# (可選) Selenium for dynamic pages
USE_SELENIUM = False
if USE_SELENIUM:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    def fetch_full_text_selenium(url, timeout=10):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(timeout)
        try:
            driver.get(url)
            html = driver.page_source
        except Exception:
            driver.quit()
            return ""
        driver.quit()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text

# ========== OCR & PDF/HTML 解析輔助函式 ==========

def parse_pdf_with_pdfplumber(pdf_bytes):
    """
    先嘗試用 pdfplumber 解析可選取文字的 PDF。
    若 PDF 是掃描，可能會得到空字串。
    """
    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        text = "\n".join(text_pages)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"[pdfplumber 解析失敗] {e}")
        return ""

def parse_pdf_with_ocr(pdf_bytes):
    """
    將 PDF 每頁轉成圖片，再用 Tesseract OCR (繁體中文) 辨識。
    """
    try:
        # 提高 DPI (解析度) 讓 OCR 更準確
        images = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as e:
        print(f"[PDF -> Image 轉換失敗] {e}")
        return ""
    
    ocr_texts = []
    for img in images:
        try:
            # 若需要中英混合，可改為 lang="chi_tra+eng"
            text = pytesseract.image_to_string(img, lang="chi_tra")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                ocr_texts.append(text)
        except Exception as e:
            print(f"[OCR 失敗] {e}")
    return "\n".join(ocr_texts)

def parse_image_with_ocr(image_bytes):
    """
    直接對圖片檔進行 OCR (繁體中文)。
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang="chi_tra")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"[影像 OCR 失敗] {e}")
        return ""

def parse_html(html_text):
    """
    解析 HTML，提取可見文字。
    """
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_full_text_requests(url, timeout=10):
    """
    透過 requests 抓取網頁內容並自動判斷類型：
    1) PDF -> 先 pdfplumber，若沒文字再 OCR
    2) 圖片 -> OCR (繁體中文)
    3) HTML -> BeautifulSoup
    4) 其他 -> 回傳空字串
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code != 200:
            print(f"[HTTP Error] 狀態碼: {resp.status_code}")
            return ""
        
        content_type = resp.headers.get("Content-Type", "").lower()
        content_bytes = resp.content

        # PDF
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = parse_pdf_with_pdfplumber(content_bytes)
            if not text.strip():
                print("[PDFplumber 無文字] 嘗試 OCR...")
                text = parse_pdf_with_ocr(content_bytes)
            return text
        
        # 圖片
        elif "image" in content_type or re.search(r"\.(png|jpe?g|gif|bmp|tif)$", url.lower()):
            return parse_image_with_ocr(content_bytes)
        
        # HTML
        elif "html" in content_type:
            html_text = resp.text
            return parse_html(html_text)
        
        else:
            print(f"[未知或不支援的 Content-Type] {content_type}")
            return ""
    
    except requests.exceptions.RequestException as e:
        print(f"[fetch_full_text_requests] 連線失敗: {url} - {e}")
        return ""
    except Exception as e:
        print(f"[fetch_full_text_requests] 無法抓取 {url} - {e}")
        return ""

def fetch_full_text(url, timeout=10):
    """
    若 robots.txt 允許，就抓取內容；可選擇 Selenium 或 requests。
    """
    if not can_crawl(url):
        print(f"[Robots] 不允許爬取: {url}")
        return ""

    if USE_SELENIUM:
        return fetch_full_text_selenium(url, timeout=timeout)
    else:
        return fetch_full_text_requests(url, timeout=timeout)

def search_duckduckgo(query, binary_labels, max_results=5):
    """
    使用 DuckDuckGo 搜尋並抓取全文，
    將「Academic Paper」的機率分數 (0~1) 存成 academic_score。
    回傳一個 list[dict]，每筆資料包含:
      - title
      - link
      - snippet
      - full_text
      - academic_score
    """
    results = []
    with DDGS() as ddgs:
        raw_results = ddgs.text(
            keywords=query,
            region="wt-wt",
            safesearch="Off",
            timelimit=None,
            max_results=max_results
        )
        for r in tqdm(raw_results, desc="處理搜尋結果"):
            title = r.get("title", "")
            link = r.get("href", "")
            snippet = r.get("body", "")

            full_text = fetch_full_text(link)
            if not full_text:
                print(f"[跳過] 無法抓取或內容為空: {link}")
                continue
            
            # 計算論文程度分數
            relate_score = get_relate_domain_score(full_text, binary_labels)

            results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "full_text": full_text,
                "academic_score": relate_score
            })
    return results

def main():
    query = input("請輸入搜尋關鍵字: ")
    if not query.strip():
        print("搜尋關鍵字不能為空！")
        return
    
    binary_labels = input("請輸入要分類的標籤，用空格隔開。例如：看漲 看跌。\n：")
    # 可選：設定搜尋結果數量
    max_results = int(input("請輸入要搜尋的結果數量 (太多筆可能導致當掉)：") or 5)
    print(f"開始搜尋: {query} ({max_results} 筆)")
    
    # 1) 搜尋並抓取全文 + 論文程度分數
    results = search_duckduckgo(query, max_results=max_results, binary_labels=binary_labels)
    if not results:
        print("沒有任何搜尋結果。")
        return
    
    # 2) 將所有結果存成 JSON 檔 (包含 academic_score)
    with open("all_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"完成！成功抓取 {len(results)} 筆結果，已儲存至 search_results.json。")

if __name__ == "__main__":
    main()