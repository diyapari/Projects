!pip install beautifulsoup4
!pip install requests
import streamlit as st
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin

# ---------------------- Functions ----------------------

def get_html(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req).read()

def scrape_ebay(keywords):
    query = '+'.join([quote_plus(k.strip()) for k in keywords])
    search_url = f"https://www.ebay.com/sch/i.html?_nkw={query}"
    html = get_html(search_url)
    soup = BeautifulSoup(html, 'html.parser')

    links = set()
    for tag in soup.find_all("a", href=True, class_="s-item__link"):
        links.add(tag.get("href"))
    return list(links)

def scrape_amazon(keywords):
    query = '+'.join([quote_plus(k.strip()) for k in keywords])
    search_url = f"https://www.amazon.com/s?k={query}"
    html = get_html(search_url)
    soup = BeautifulSoup(html, 'html.parser')

    links = set()
    for tag in soup.select("a.a-link-normal.s-no-outline"):
        href = tag.get("href")
        if href:
            full_url = urljoin("https://www.amazon.com", href)
            links.add(full_url)
    return list(links)

def scrape_tradeindia(keywords):
    query = '+'.join([quote_plus(k.strip()) for k in keywords])
    search_url = f"https://www.tradeindia.com/search.html?keyword={query}"
    html = get_html(search_url)
    soup = BeautifulSoup(html, 'html.parser')

    links = set()
    for tag in soup.find_all("a", href=True):
        url = tag.get("href")
        if url and "product" in url.lower():
            full_url = urljoin("https://www.tradeindia.com", url)
            links.add(full_url)
    return list(links)

def scrape_aliexpress(keywords):
    query = '+'.join([quote_plus(k.strip()) for k in keywords])
    search_url = f"https://www.aliexpress.com/wholesale?SearchText={query}"
    html = get_html(search_url)
    soup = BeautifulSoup(html, 'html.parser')

    links = set()
    for tag in soup.find_all("a", href=True):
        url = tag.get("href")
        if url and "/item/" in url:
            full_url = urljoin("https://www.aliexpress.com", url)
            links.add(full_url)
    return list(links)

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Dropship Finder", layout="centered", page_icon="🛒")

# Banner image (you can change the URL or use st.image with your own image)
st.markdown("""
    <div style='text-align:center'>
        <img src='https://cdn-icons-png.flaticon.com/512/891/891419.png' width='120'/>
        <h1 style='color:#4A4A4A;'>🌐 Dropship Finder</h1>
        <p style='color:gray;font-size:16px;'>Find product links from major e-commerce platforms.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("🔍 Search Configuration")
site = st.sidebar.selectbox("Select Platform", ["eBay", "Amazon", "TradeIndia", "AliExpress"])
user_input = st.sidebar.text_input("Enter keywords (comma-separated)", placeholder="e.g., shoes, bag")

if st.sidebar.button("🔎 Search"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some search keywords.")
    else:
        keywords = user_input.strip().split(',')

        with st.spinner("Scraping product links..."):
            if site == "eBay":
                links = scrape_ebay(keywords)
            elif site == "Amazon":
                links = scrape_amazon(keywords)
            elif site == "TradeIndia":
                links = scrape_tradeindia(keywords)
            elif site == "AliExpress":
                links = scrape_aliexpress(keywords)
            else:
                links = []

        st.subheader(f"🔗 Results from {site}")
        if links:
            st.success(f"✅ Found {len(links)} product links!")
            for link in links[:20]:  # Show top 20
                st.markdown(f"🔸 [View Product]({link})")
        else:
            st.error("❌ No product links found. Try different keywords.")

# Optional footer
st.markdown("<hr><div style='text-align:center;color:gray;'>Made with using Python & Streamlit</div>", unsafe_allow_html=True)
