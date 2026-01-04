import streamlit as st
import os
import requests
import io
import time
from bs4 import BeautifulSoup
from apify_client import ApifyClient
from google import genai  
from google.genai import types 
from dotenv import load_dotenv
from pypdf import PdfReader 
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Load variables from .env if running locally
load_dotenv()

config = types.GenerateContentConfig(
        temperature=0.1,    # Closer to 0 is more deterministic
        top_p=0.95,         # Helps focus on high-probability tokens
        top_k=35,           # Limits the vocabulary pool
        max_output_tokens=15000,
    )

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Analyst", layout="wide")

APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_KEY and APIFY_TOKEN:
    client = genai.Client(api_key=GEMINI_KEY)
    apify_client = ApifyClient(APIFY_TOKEN)
else:
    st.error("Missing API Keys! Please set GEMINI_API_KEY and APIFY_TOKEN in your environment.")

# --- 2. SCRAPING LOGIC (Screener + Apify) ---

def extract_transcript_links(html_content):
    """Parses Screener.in HTML specifically for transcript PDF links."""
    soup = BeautifulSoup(html_content, 'html.parser')
    transcript_links = []
    # Find all list items representing a concall entry
    list_items = soup.find_all('li', class_='flex-wrap-420')

    for li in list_items:
        # Look for <a> tag with class 'concall-link' where text is 'Transcript'
        transcript_anchor = li.find('a', class_='concall-link', string='Transcript')
        if transcript_anchor and transcript_anchor.has_attr('href'):
            link = transcript_anchor['href']
            # Ensure links are absolute
            full_link = link if link.startswith('http') else f"https://www.screener.in{link}"
            transcript_links.append(full_link)
    
    return transcript_links

def get_financial_data(ticker):
    """
    Fetches Ratios via Apify and Transcript Links via direct Scraping.
    """
    try:
        # A. Fetch Ratios via Apify
        company_url = f"https://www.screener.in/company/{ticker}/consolidated/"
        run_input = { "mode": "getstockdetails", "url": company_url }
        
        # Call Apify for the structured financial data (ratios)
        run = apify_client.actor("shashwattrivedi/screener-in").call(run_input=run_input)
        
        ratios = {}
        for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            ratios = item.get("ratios", item)
            break 
        
        # B. Fetch Transcript Links via BeautifulSoup (since actor might miss them)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(company_url, headers=headers, timeout=10)
        links = []
        if response.status_code == 200:
            links = extract_transcript_links(response.text)
            
        return ratios, links
    except Exception as e:
        st.error(f"Data Acquisition Error: {e}")
        return None, None

# --- 3. PDF PROCESSING ---

def read_transcript(url):
    """Downloads and extracts text from a PDF."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        
        with io.BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = ""
            num_pages = len(reader.pages)
            # Limit to 10 pages for speed and memory efficiency
            for i in range(min(10, num_pages)):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    text += f"\n--- PAGE {i+1} ---\n{page_text}"
            
            return text if text.strip() else "[Scanned PDF/No Text]"
            
    except Exception as e:
        return f"Could not read PDF: {str(e)}"

# --- 4. AI ANALYSIS ---

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=10, max=60),
    stop=stop_after_attempt(3)
)
def generate_analysis(prompt):
    """Uses Gemini 2.5 Flash for final audit."""
    return client.models.generate_content(
        model="gemini-2.5-flash", # Note: Ensure model name is correct for current API
        contents=prompt,
        config=config
    )

###############################################################


from fpdf import FPDF

def create_pdf(text, company_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Forensic Audit Report: {company_name}", ln=True, align='C')
    pdf.ln(10) # Line break
    
    # Body Text
    pdf.set_font("Arial", size=11)
    # multi_cell automatically handles line wrapping and page breaks
    pdf.multi_cell(0, 10, txt=text)
    
    # Return the PDF as a byte string
    return pdf.output(dest='S') # 'S' returns the document as a string (bytes in fpdf2)

import requests
from bs4 import BeautifulSoup

def get_company_ratios(ticker):
    """
    Fetches the main ratios from the 'top-ratios' section of Screener.in
    """
    url = f"https://www.screener.in/company/{ticker}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
            return {}

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Locate the specific list by its ID
        ratios_container = soup.find('ul', id='top-ratios')
        if not ratios_container:
            print("Could not find the 'top-ratios' section.")
            return {}

        ratios_data = {}
        
        # Iterate through each list item (li) in the ratios section
        for li in ratios_container.find_all('li', class_='flex'):
            # Extract the ratio name
            name_tag = li.find('span', class_='name')
            # Extract the value (which may contain currency symbols and nested spans)
            value_tag = li.find('span', class_='value')
            
            if name_tag and value_tag:
                name = name_tag.get_text(strip=True)
                # We use a space separator to keep units like 'Cr.' separate from the number
                value = value_tag.get_text(" ", strip=True).replace('\n', '').replace('  ', ' ')
                ratios_data[name] = value
                print(ratios_data)
        return ratios_data

    except Exception as e:
        print(f"Error scraping ratios: {e}")
        return {}

# Example Usage

###############################################
# --- 5. MAIN APP INTERFACE ---

st.title("Reliable AI Stock Research Analysis tool by Adithya B M")
st.caption("Considering Fundamental financial data of the company + Concall Transcripts + Cashflow, shareholding pattern, profit&loss ")
st.caption("The procedure of fundamentally analysing the stock is given by sebi registered research analyst, But this is not any investment advice or anything, you can use it as source of info for our investment decisions.")
ticker = st.text_input("Enter any Indian Listed company ticker symbol (Eg:-INFY,TCS,RELIANCE):").upper()

if st.button("Start Deep Research"):
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        with st.status("Stock analysis in Progress using the management conference call and other many financal data", expanded=True) as status:
            # Step 1: Data Acquisition
            status.write("Scraping Fundamental financial data")
            main_ratios = get_company_ratios(ticker)
            status.write("Gathering Financial Ratios from trusted source which is screener.in & Concall trascript Links...")
            ratios, links = get_financial_data(ticker)
            
            if not ratios:
                st.error("Failed to fetch financial data.")
                st.stop()
            
            # Step 2: PDF Parsing
            all_transcripts_text = ""
            if links:
                # We only take the top 2 most recent transcripts to stay within Gemini's context window
                recent_links = links[:3] 
                for i, link in enumerate(recent_links):
                    status.write(f"Downloading & Parsing Transcript {i+1} of {len(recent_links)}...")
                    all_transcripts_text += f"\n--- TRANSCRIPT {i+1} SOURCE ---\n" + read_transcript(link)
            else:
                status.write("No transcripts found. Proceeding with Ratio analysis only.")

            # Step 3: AI Analysis
            status.write("Analysing all the data with the standard procedure given by a RA using our AI model.")
            
            # Constructing the Audit Prompt
            audit_prompt =f"""
             
            Company: {ticker}
            
            DATA:
            - Fundamental Ratios and data: {main_ratios}
            - Ratios: {ratios}
            - Transcript Snippets: {all_transcripts_text[:165000]}
            
             

             Assume the role of an experienced fund manager with over 20 years in the stock market. Analyze the company {ticker} and provide a detailed report covering the following aspects:
	1.	Company History and Management:
	•	Outline the company’s background, including its inception, evolution, and key milestones.
	•	Evaluate the credibility and experience of the management team, highlighting their track record in the industry.
	2.	Business Model:
	•	Explain how the company generates revenue, detailing its primary objectives. Even people with different fields can easily understand with examples of products or services and target markets.
	•	Describe the cost structure and key factors influencing profitability.
	•	Ensure that the business model is straightforward enough that a 15-year-old could understand it, aligning with Warren Buffett’s principle that if a business is too complex to understand, it may not be a suitable investment.
	3.	Competitive Advantage (Moat):
	•	Identify the company’s unique selling propositions and factors that differentiate it from competitors. ￼
	•	Discuss any barriers to entry that protect the company’s market position.
	4.	Future Plans and Growth Prospects:
	•	Summarize the company’s strategic initiatives, such as planned product launches, market expansions, or mergers and acquisitions.
	•	Analyze the industry’s growth rate and assess the company’s potential to outpace industry growth.
	5.	Financial Health:
	•	Review key financial metrics, including revenue and profit trends over the past five years.
	•	Evaluate the strength of the balance sheet by examining assets, liabilities, and equity.
	•	Analyze cash flow statements to determine the company’s ability to generate and utilize cash effectively.
	6.	Valuation:
	•	Calculate and interpret valuation ratios such as the Price-to-Earnings (P/E) ratio, Price-to-Book (P/B) ratio, and any other relevant metrics.
	•	Compare these ratios to industry averages to assess whether the company’s stock is overvalued, undervalued, or fairly priced.
	7.	Pros and Cons:
	•	Highlight the strengths and opportunities that make the company a compelling investment.
	•	Discuss potential weaknesses and threats that could impact the company’s performance.
	8.	Five-Year Financial Projections:
	•	Provide forecasted financial statements, including projected revenues, expenses, and earnings per share (EPS) for the next five years.
	•	Outline the assumptions underlying these projections and discuss potential risks to achieving them.
    9.  Give the risks to invest in this share and points in breif detail why would it go down and why would it go up.
    10. Give a fair price for this stock but indicate that not to invest based on this price, since the price could not be accurate.
Ensure that the analysis is thorough, data-driven, and presented in a manner understandable to a layperson. Use examples where appropriate to illustrate key points. If certain information is unavailable, indicate this clearly in the report if data is not found in transcripts or given data, try to find in the google web but never hallucinate.

"""
            
            try:
                response = generate_analysis(audit_prompt)
                status.update(label="Audit Complete!", state="complete")
                st.divider()
                st.markdown(response.text)
                # clean_text = response.text.replace("₹", "Rs.")
                # pdf_bytes = create_pdf(clean_text, ticker)

                # # Create the Download Button
                # st.download_button(
                #     label="📥 Download Audit Report as PDF",
                #     data=pdf_bytes,
                #     file_name=f"{ticker}_Forensic_Audit.pdf",
                #     mime="application/pdf"
                # )
            except Exception as e:
                st.error(f" Error:  {e}")