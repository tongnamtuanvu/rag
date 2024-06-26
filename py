import streamlit as st
import openai
import plotly.express as px
from torch import cuda
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain import OpenAI
import pandas as pd
import csv
import json
import difflib

# Set OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Determine device
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Initialize embedding function and model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OpenAI(base_url="https://direct-piranha-hideously.ngrok-free.app/v1/", api_key="not-needed")

# Initialize Chroma vectorstore
chroma_db_directory = "./chroma_db_nccn"
vector_db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding_function)

# Define keyword lists
keywords_software = [
    "Software development", "Software engineering", "Software solutions", "Software architecture", "Software testing",
    "Software design", "Software deployment", "Software lifecycle", "Software maintenance", "Software integration",
    "Software quality assurance", "Software project management", "Software requirements", "Software security",
    "Software performance", "Software optimization", "Software automation", "Software customization", "Software upgrades",
    "Software migration", "Software scalability", "Software support", "Software documentation", "Software configuration",
    "Software version control", "Software frameworks", "Software libraries", "Software tools", "Software platforms",
    "Software development kits (SDKs)", "Agile software development", "Software methodology", "Software processes",
    "Software deployment pipeline", "Software repository", "Software build", "Software release management", "Software debugging",
    "Software refactoring", "Software innovation", "Software patents", "Software licensing", "Software compliance", "Software audit",
    "Software validation", "Software verification", "Software compatibility", "Software user interface", "Software user experience",
    "Software prototyping", "Software scalability", "Software architecture patterns", "Software microservices", "Software containerization",
    "Software virtualization", "Software cloud computing", "Software SaaS (Software as a Service)", "Software PaaS (Platform as a Service)",
    "Software IaaS (Infrastructure as a Service)", "Software DevOps", "Software continuous integration", "Software continuous delivery",
    "Software CI/CD pipelines", "Software configuration management", "Software monitoring", "Software incident management", "Software disaster recovery",
    "Software data management", "Software data analysis", "Software big data", "Software machine learning", "Software artificial intelligence",
    "Software deep learning", "Software natural language processing", "Software computer vision", "Software robotics", "Software IoT (Internet of Things)",
    "Software blockchain", "Software cybersecurity", "Software encryption", "Software authentication", "Software authorization",
    "Software API development", "Software RESTful services", "Software GraphQL", "Software web development", "Software mobile development",
    "Software desktop applications", "Software embedded systems", "Software game development", "Software VR (Virtual Reality)",
    "Software AR (Augmented Reality)", "Software multimedia", "Software audio processing", "Software video processing", "Software graphics",
    "Software simulations", "Software health informatics", "Software fintech", "Software edtech"
]

keywords_marketing = [
    "Digital marketing", "Content marketing", "SEO (Search Engine Optimization)", "PPC (Pay-Per-Click)", "Social media marketing",
    "Email marketing", "Affiliate marketing", "Influencer marketing", "Brand management", "Marketing strategy",
    "Market research", "Customer segmentation", "Target audience", "Marketing analytics", "Conversion rate optimization",
    "Lead generation", "Customer retention", "Product marketing", "Event marketing", "Mobile marketing",
    "Video marketing", "Viral marketing", "Growth hacking", "B2B marketing", "B2C marketing",
    "Brand awareness", "Customer journey", "Marketing funnel", "Marketing automation", "Landing pages",
    "A/B testing", "User experience (UX)", "Inbound marketing", "Outbound marketing", "Guerrilla marketing",
    "Native advertising", "Programmatic advertising", "Retargeting", "Sponsorship", "Webinars",
    "Podcast marketing", "Local SEO", "Online reputation management", "Crisis management", "Customer relationship management (CRM)",
    "Public relations (PR)", "Corporate communications", "Sales enablement", "Marketing collateral", "Marketing campaigns",
    "E-commerce marketing", "Retail marketing", "Direct mail marketing", "Print advertising", "Telemarketing",
    "Outdoor advertising", "Radio advertising", "TV advertising", "Display advertising", "Remarketing",
    "Behavioral targeting", "Contextual targeting", "Geo-targeting", "Demographic targeting", "Psychographic targeting",
    "Market positioning", "Unique selling proposition (USP)", "Value proposition", "Brand storytelling", "Brand voice",
    "Brand loyalty", "Customer feedback", "Customer advocacy", "Customer insights", "Competitive analysis",
    "SWOT analysis", "Marketing mix", "4 Ps of marketing", "7 Ps of marketing", "Marketing budget",
    "Marketing ROI (Return on Investment)", "Customer acquisition cost (CAC)", "Lifetime value (LTV)", "Churn rate",
    "Market penetration", "Market expansion", "Co-branding", "Cause marketing", "Sponsorship marketing",
    "Integrated marketing communications (IMC)", "Marketing ethics", "Consumer behavior", "Green marketing",
    "Experiential marketing", "Omnichannel marketing", "Multichannel marketing", "Interactive marketing",
    "Proximity marketing", "Conversational marketing", "Voice search optimization"
]

keywords_finance = [
    "Investment banking", "Asset management", "Wealth management", "Financial planning", "Risk management",
    "Corporate finance", "Personal finance", "Financial analysis", "Stock market", "Bonds",
    "Mutual funds", "Exchange-traded funds (ETFs)", "Hedge funds", "Private equity", "Venture capital",
    "Real estate investment", "Portfolio management", "Retirement planning", "Estate planning", "Insurance",
    "Financial advisory", "Financial modeling", "Financial forecasting", "Budgeting", "Cash flow management",
    "Credit analysis", "Credit risk", "Market risk", "Operational risk", "Liquidity management",
    "Capital budgeting", "Capital structure", "Financial ratios", "Valuation", "Mergers and acquisitions (M&A)",
    "Initial public offering (IPO)", "Debt financing", "Equity financing", "Dividend policy", "Share buybacks",
    "Financial regulation", "Compliance", "Tax planning", "Accounting standards", "Auditing",
    "Financial reporting", "Earnings management", "Cost of capital", "Working capital management", "Treasury management",
    "Foreign exchange", "Commodities trading", "Derivatives", "Options", "Futures",
    "Swaps", "Forward contracts", "Financial derivatives", "Credit default swaps", "Interest rate swaps",
    "Hedging strategies", "Speculation", "Arbitrage", "Behavioral finance", "Quantitative finance",
    "Algorithmic trading", "High-frequency trading", "Blockchain technology", "Cryptocurrency", "Bitcoin",
    "Fintech", "Insurtech", "Regtech", "Robo-advisors", "Digital banking",
    "Mobile payments", "Peer-to-peer lending", "Crowdfunding", "Microfinance", "Islamic finance",
    "Green finance", "Sustainable finance", "Impact investing", "Socially responsible investing (SRI)", "Environmental, social, and governance (ESG) criteria",
    "Economic indicators", "Interest rates", "Inflation", "Gross domestic product (GDP)", "Monetary policy",
    "Fiscal policy", "Central banking", "Federal Reserve", "European Central Bank (ECB)", "Bank of Japan (BOJ)",
    "International Monetary Fund (IMF)", "World Bank", "Credit rating agencies", "Sovereign debt", "Financial crisis",
    "Economic recession", "Market volatility", "Financial innovation", "Digital assets"
]

keywords_business = [
    "Business strategy", "Business development", "Business model", "Entrepreneurship", "Startup",
    "Small business", "Corporate strategy", "Business planning", "Market analysis", "Competitive analysis",
    "SWOT analysis", "Business growth", "Scalability", "Revenue streams", "Cost structure",
    "Value proposition", "Business operations", "Operational efficiency", "Process improvement", "Supply chain management",
    "Logistics", "Inventory management", "Quality control", "Lean management", "Six Sigma",
    "Project management", "Program management", "Change management", "Organizational behavior", "Human resources",
    "Talent management", "Recruitment", "Employee retention", "Leadership development", "Team building",
    "Corporate culture", "Corporate governance", "Ethics", "Corporate social responsibility (CSR)", "Sustainability",
    "Innovation management", "Product development", "Product lifecycle management", "Brand management", "Customer relationship management (CRM)",
    "Customer experience", "Customer loyalty", "Customer feedback", "Sales strategy", "Sales operations",
    "Salesforce management", "Sales performance", "Marketing strategy", "Market segmentation", "Target market",
    "Pricing strategy", "Promotions", "Advertising", "Digital marketing", "Social media marketing",
    "Content marketing", "Public relations", "Investor relations", "Financial management", "Budgeting",
    "Financial forecasting", "Accounting", "Cost management", "Profitability analysis", "Cash flow management",
    "Financial statements", "Balance sheet", "Income statement", "Cash flow statement", "Tax planning",
    "Compliance", "Risk management", "Insurance", "Legal compliance", "Intellectual property",
    "Contract management", "Business law", "Negotiation skills", "Conflict resolution", "Decision making",
    "Problem solving", "Strategic thinking", "Critical thinking", "Business communication", "Presentation skills",
    "Networking", "Time management", "Productivity", "Performance metrics", "Benchmarking",
    "Key performance indicators (KPIs)", "Data analytics", "Business intelligence", "Big data", "Data-driven decision making"
]

keywords_food = [
    "Healthy eating", "Organic food", "Natural ingredients", "Farm-to-table", "Food safety",
    "Food quality", "Food processing", "Food packaging", "Food preservation", "Food additives",
    "Nutrition", "Balanced diet", "Vitamins", "Minerals", "Proteins",
    "Carbohydrates", "Fats", "Calories", "Superfoods", "Functional foods",
    "Dietary supplements", "Vegan", "Vegetarian", "Gluten-free", "Dairy-free",
    "Allergy-friendly", "Food allergies", "Food intolerances", "Sustainable food", "Food waste",
    "Food recycling", "Composting", "Zero waste", "Food security", "Food distribution",
    "Food supply chain", "Local food", "Seasonal food", "Ethnic cuisine", "International cuisine",
    "Gourmet food", "Street food", "Fast food", "Convenience food", "Frozen food",
    "Ready-to-eat meals", "Home cooking", "Meal prep", "Recipe development", "Cooking techniques",
    "Baking", "Grilling", "Roasting", "Steaming", "Boiling",
    "Sautéing", "Frying", "Sous-vide", "Fermentation", "Pickling",
    "Food trends", "Plant-based food", "Meat alternatives", "Insect protein", "Lab-grown meat",
    "Food innovation", "Food tech", "Culinary arts", "Food photography", "Food blogging",
    "Food marketing", "Food branding", "Food labeling", "Nutritional labeling", "Ingredient sourcing",
    "Food import/export", "Food retail", "Supermarkets", "Grocery stores", "Farmers markets",
    "Online food delivery", "Meal kits", "Food trucks", "Catering", "Restaurant industry",
    "Hospitality industry", "Food tourism", "Culinary travel", "Food festivals", "Food and beverage",
    "Beverage industry", "Alcoholic beverages", "Non-alcoholic beverages", "Coffee industry", "Tea industry",
    "Dairy industry", "Bakery industry", "Confectionery", "Snacks", "Pet food"
]

keywords_fmcg = [
    "Consumer goods", "Fast-moving consumer goods", "FMCG market", "Retail industry", "Product innovation",
    "Brand management", "Product lifecycle", "Market segmentation", "Consumer behavior", "Market research",
    "Sales strategy", "Distribution channels", "Supply chain management", "Inventory management", "Demand forecasting",
    "Logistics", "Warehousing", "Retail merchandising", "Category management", "Product assortment",
    "Pricing strategy", "Promotional strategy", "Marketing campaigns", "Advertising", "Digital marketing",
    "Social media marketing", "Content marketing", "Influencer marketing", "Brand loyalty", "Customer engagement",
    "Customer experience", "Customer satisfaction", "Consumer insights", "Retail analytics", "Data-driven decision making",
    "Point of sale (POS)", "E-commerce", "Omnichannel retail", "Online grocery", "Home delivery",
    "Click and collect", "Retail technology", "Mobile commerce", "Consumer trends", "Sustainable packaging",
    "Eco-friendly products", "Health and wellness", "Organic products", "Natural products",
    "Private label", "Store brands", "Product differentiation", "Value proposition", "Brand equity",
    "Market share", "Competitive analysis", "SWOT analysis", "Product launch", "Product positioning",
    "Retail partnerships", "Franchising", "Licensing", "Global expansion", "Emerging markets",
    "Regulatory compliance", "Quality control", "Product safety", "Food safety", "Shelf life",
    "Packaging design", "Labeling", "Nutritional information", "Consumer protection", "Intellectual property",
    "Trade marketing", "In-store promotions", "Sampling campaigns", "Consumer promotions", "Retail displays",
    "Planograms", "Shopper marketing", "Customer loyalty programs", "CRM (Customer Relationship Management)", "Loyalty cards",
    "Market penetration", "Product penetration", "Retail audits", "Consumer panels", "Household panels",
    "Product reviews", "Consumer feedback", "Brand recall", "Top-of-mind awareness", "Advertising recall",
    "Market saturation", "Product obsolescence", "Seasonal products", "Impulse buying", "Cross-selling",
    "Up-selling", "Market dynamics", "Trade shows", "Retail conferences", "Industry reports"
]

keywords_bank = [
    "Retail banking", "Commercial banking", "Investment banking", "Corporate banking", "Private banking",
    "Wealth management", "Asset management", "Risk management", "Credit risk", "Market risk",
    "Operational risk", "Liquidity risk", "Compliance", "Regulatory compliance", "Anti-money laundering (AML)",
    "Know Your Customer (KYC)", "Basel III", "Dodd-Frank Act", "Banking regulations", "Capital adequacy",
    "Tier 1 capital", "Tier 2 capital", "Stress testing", "Credit scoring", "Loan origination",
    "Loan underwriting", "Mortgage lending", "Personal loans", "Auto loans", "Credit cards",
    "Debit cards", "Prepaid cards", "Online banking", "Mobile banking", "Digital banking",
    "Banking apps", "Financial technology (Fintech)", "Blockchain", "Cryptocurrency", "Digital payments",
    "Peer-to-peer payments", "Mobile wallets", "Contactless payments", "Electronic funds transfer (EFT)", "Automated Clearing House (ACH)",
    "Wire transfers", "SWIFT", "Real-time gross settlement (RTGS)", "Payment gateways", "Merchant services",
    "Point of sale (POS)", "ATM networks", "Branch banking", "Banking customer service", "Customer relationship management (CRM)",
    "Customer experience", "Customer satisfaction", "Customer loyalty", "Customer insights", "Banking analytics",
    "Data analytics", "Big data", "Artificial intelligence (AI)", "Machine learning", "Predictive analytics",
    "Fraud detection", "Cybersecurity", "Information security", "Data privacy", "Identity theft",
    "Banking software", "Core banking systems", "Banking infrastructure", "Open banking", "API banking",
    "Banking as a Service (BaaS)", "Robo-advisors", "Wealthtech", "Insurtech", "Regtech",
    "Financial inclusion", "Microfinance", "Islamic banking", "Green banking", "Sustainable finance",
    "Socially responsible banking", "Corporate social responsibility (CSR)", "Community banking", "Credit unions", "Cooperative banking",
    "Savings and loan associations", "Treasury management", "Cash management", "Foreign exchange (Forex)", "Trade finance",
    "Letters of credit", "Documentary collections", "Supply chain finance", "Factoring", "Forfaiting",
    "Project finance", "Infrastructure finance", "Syndicated loans", "Leveraged finance", "Mergers and acquisitions (M&A)"
]

keywords_education = [
    "Education technology (EdTech)", "Online learning", "E-learning", "Blended learning", "Distance education",
    "Virtual classrooms", "Learning management systems (LMS)", "Educational apps", "Gamification in education", "Adaptive learning",
    "Artificial intelligence in education", "Personalized learning", "Student engagement", "Digital literacy", "STEM education",
    "STEAM education", "Project-based learning", "Flipped classroom", "Competency-based education", "Collaborative learning",
    "21st-century skills", "Critical thinking", "Problem-solving skills", "Creativity in education", "Communication skills",
    "Educational policy", "Educational leadership", "School administration", "Curriculum development", "Instructional design",
    "Teacher training", "Professional development", "Inclusive education", "Special education", "Early childhood education",
    "Primary education", "Secondary education", "Higher education", "Adult education", "Vocational training",
    "Lifelong learning", "Continuing education", "Teacher-student ratio", "Student assessment", "Standardized testing",
    "Formative assessment", "Summative assessment", "Educational research", "Learning theories", "Cognitive development",
    "Behavioral psychology", "Social-emotional learning", "Educational psychology", "Classroom management", "School safety",
    "Bullying prevention", "Parental involvement", "Community engagement", "Educational equity", "Achievement gap",
    "Learning disabilities", "Gifted education", "Bilingual education", "Multicultural education", "Language acquisition",
    "Reading comprehension", "Literacy programs", "Numeracy programs", "Science education", "Mathematics education",
    "Technology in the classroom", "Mobile learning", "Educational games", "Interactive whiteboards", "Digital textbooks",
    "Open educational resources (OER)", "Massive open online courses (MOOCs)", "Peer learning", "Mentorship programs",
    "School funding", "Educational grants", "Scholarships", "Student loans", "Tuition fees",
    "Educational infrastructure", "School facilities", "Library services", "Educational partnerships", "Workforce development",
    "Career counseling", "Job placement services", "Internships", "Apprenticeships", "Alumni networks",
    "Global education", "International students", "Study abroad programs", "Exchange programs", "Education policy reform"
]

all_keywords = {
    "Software": keywords_software,
    "Marketing": keywords_marketing,
    "Finance": keywords_finance,
    "Business": keywords_business,
    "Food": keywords_food,
    "FMCG": keywords_fmcg,
    "Bank": keywords_bank,
    "Education": keywords_education
}

@st.cache_data
def load_data(file):
    try:
        if isinstance(file, str):
            file_type = file.split('.')[-1]
            if file_type in ["csv", "txt"]:
                return pd.read_csv(file)
            elif file_type in ["xls", "xlsx"]:
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file type")
        else:
            file_type = file.name.split('.')[-1]
            if file_type in ["csv", "txt"]:
                sample = file.read(1024).decode()
                file.seek(0)
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except csv.Error:
                    delimiter = ','  # Default to comma if detection fails
                return pd.read_csv(file, delimiter=delimiter)
            elif file_type in ["xls", "xlsx"]:
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file type")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def handle_online_search(prompt):
    web_links = [f"https://www.google.com/search?q={prompt.replace(' ', '+')}"]
    loader = WebBaseLoader(web_links)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(all_splits, embedding_function)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                  return_source_documents=True)

    chat_history = []

    result = chain({"question": prompt, "chat_history": chat_history})
    keywords = extract_keywords(result['answer'])

    st.session_state.company_keywords = keywords
    translated_result = translate_to_vietnamese(result['answer'])

    return translated_result

def translate_to_vietnamese(text):
    translation_prompt = f"Translate the following text to Vietnamese, keeping technical terms in English:\n\n{text}"
    translation_result = llm(translation_prompt)
    return translation_result.strip()

def extract_keywords(text):
    keywords = set()
    for category, keyword_list in all_keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in text.lower():
                keywords.add(keyword)
    return list(keywords)

def handle_data_analysis(prompt, data):
    text_data = data.to_string()
    full_prompt = f"{prompt}\n\nData:\n{text_data}"
    result = generate_full_response(llm, full_prompt)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result

    if 'text' in result:
        return result['text']
    else:
        return "No 'text' key found in the result"

def handle_offline_interaction(prompt, data):
    text_data = data.to_string()
    preprompt = st.sidebar.text_area("Adjust Pre-prompt", f"Bạn là chuyên gia trong lĩnh vực, hãy đưa ra phân tích và insight cho dữ liệu này, doanh nghiệp có thể làm gì với dữ liệu này cũng như dữ liệu này giúp ích gì cho doanh nghiệp:\n")
    full_prompt = f"{preprompt}\n {prompt}\n{text_data}"
    result = generate_full_response(llm, full_prompt)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result

    if 'text' in result:
        return result['text']
    else:
        return "No 'text' key found in the result"

def generate_full_response(llm, prompt, max_tokens=1000):
    response = llm(prompt, max_tokens=max_tokens)
    if isinstance(response, str):
        result = response
    else:
        result = response.get('choices', [])[0].get('text', '')

    while isinstance(response, dict) and response.get('choices', [])[0].get('finish_reason') != 'stop':
        prompt = result
        response = llm(prompt, max_tokens=max_tokens)
        if isinstance(response, str):
            result += response
        else:
            result += response.get('choices', [])[0].get('text', '')

    return result

def plot_overview(data, plot_types):
    numeric_data = data.select_dtypes(include=['number'])
    categorical_data = data.select_dtypes(include=['object', 'category'])

    if numeric_data.empty and categorical_data.empty:
        st.error("No data available for plotting.")
        return

    if not numeric_data.empty:
        if "Histogram" in plot_types:
            fig = px.histogram(numeric_data)
            st.write(fig)
        if "Scatter" in plot_types and len(numeric_data.columns) > 1:
            fig = px.scatter(numeric_data, x=numeric_data.columns[0], y=numeric_data.columns[1])
            st.write(fig)
        if "Box" in plot_types:
            fig = px.box(numeric_data)
            st.write(fig)
        if "Line" in plot_types and len(numeric_data.columns) > 1:
            fig = px.line(numeric_data, x=numeric_data.columns[0], y=numeric_data.columns[1])
            st.write(fig)

    if not categorical_data.empty:
        if "Bar" in plot_types:
            fig = px.bar(categorical_data)
            st.write(fig)
        if "Pie" in plot_types:
            for col in categorical_data.columns:
                fig = px.pie(categorical_data, names=col)
                st.write(fig)

def plot_column_analysis(data, column, plot_types):
    if data[column].dtype in ['int64', 'float64']:
        if "Histogram" in plot_types:
            fig = px.histogram(data, x=column)
            st.write(fig)
        if "Scatter" in plot_types and len(data.columns) > 1:
            fig = px.scatter(data, x=column, y=data.columns[1])
            st.write(fig)
        if "Box" in plot_types:
            fig = px.box(data, x=column)
            st.write(fig)
        if "Line" in plot_types and len(data.columns) > 1:
            fig = px.line(data, x=column, y=data.columns[1])
            st.write(fig)
    elif data[column].dtype in ['object', 'category']:
        if "Bar" in plot_types:
            fig = px.bar(data, x=column)
            st.write(fig)
        if "Pie" in plot_types:
            fig = px.pie(data, names=column)
            st.write(fig)
    else:
        st.error(f"Column '{column}' has an unsupported data type for plotting.")

def extract_plotting_info(prompt):
    plot_keywords = ["vẽ chart", "vẽ biểu đồ", "biểu đồ", "chart", "vẽ"]
    chart_types = ["Histogram", "Scatter", "Box", "Line", "Bar", "Pie"]
    plot_types = []
    for keyword in plot_keywords:
        if keyword in prompt.lower():
            for chart_type in chart_types:
                if chart_type.lower() in prompt.lower():
                    plot_types.append(chart_type)
            if not plot_types:
                return True, None
            return True, plot_types
    return False, []

def display_company_keywords():
    if "company_keywords" in st.session_state:
        keywords_html = " ".join(
            [f"<button style='display:inline-block; border:1px solid black; padding:10px; margin-right:10px; background-color:#f0f0f0;' disabled>{keyword}</button>" for keyword in st.session_state.company_keywords if len(keyword) <= 53])
        st.sidebar.markdown(keywords_html, unsafe_allow_html=True)

# Main function
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <img src="/Users/vutong/Downloads/metricity.png" width="50" style="margin-right: 10px;">
            <h1>METRICITY</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    display_company_keywords()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào bạn, bạn đến từ công ty nào?"}]
    st.session_state.search_mode = False
    st.session_state.show_sample_data = False
    st.session_state.company_validated = False
    st.session_state.file_validated = False

for message in st.session_state.messages:
    if isinstance(message, dict) and "role" in message:
        with st.chat_message(message["role"]):
            st.write(message.get("content", "No content"))
    else:
        st.error("Invalid message format.")

prompt = st.chat_input("Your question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.messages[-2]["role"] == "assistant" and \
            st.session_state.messages[-2]["content"].startswith("Xin chào bạn"):
        st.session_state.company = prompt
        st.session_state.search_mode = True
        result = handle_online_search(prompt)
        st.session_state.search_mode = False

        st.session_state.messages.append({"role": "assistant", "content": result})

    elif st.session_state.search_mode:
        result = handle_online_search(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result})
    else:
        if "selected_data" in st.session_state and st.session_state.selected_data is not None:
            result = handle_offline_interaction(prompt, st.session_state.selected_data, st.session_state.company_keywords)
        else:
            result = handle_offline_interaction(prompt, pd.DataFrame(), st.session_state.company_keywords)
        st.session_state.messages.append({"role": "assistant", "content": result})

with st.sidebar:
    st.markdown("---")
    if st.checkbox("Search Online"):
        st.session_state.search_mode = True
    else:
        st.session_state.search_mode = False

    display_company_keywords()

with st.sidebar:
    st.markdown("### Company")
    company_name = st.text_input("Enter company name:")
    if st.button("Validate Company"):
        st.session_state.company = company_name
        st.session_state.search_mode = True
        search_result = handle_online_search(company_name)
        st.session_state.search_mode = False

        st.session_state.messages.append({"role": "assistant", "content": search_result})

        display_company_keywords()

with st.sidebar:
    st.markdown("---")
    uploaded_file = st.file_uploader("File for analysis", type=["csv", "xlsx", "txt"])
    if uploaded_file is not None:
        try:
            st.session_state.selected_data = load_data(uploaded_file)
            st.session_state.show_sample_data = True
        except ValueError as e:
            st.error(f"Error: {e}")

    if "selected_data" in st.session_state and st.session_state.selected_data is not None and not st.session_state.selected_data.empty:
        if st.session_state.show_sample_data:
            st.dataframe(st.session_state.selected_data)

        if st.button("Validate"):
            st.session_state.file_validated = True

        if st.session_state.file_validated:
            columns = ["All"] + st.session_state.selected_data.columns.tolist()
            selected_column = st.radio("Select column", columns, index=0, horizontal=True)
            st.session_state.selected_column = selected_column

if "selected_data" not in st.session_state or st.session_state.selected_data is None:
    st.write("No data selected or data is empty.")

if prompt and st.session_state.file_validated:
    plot_detected, plot_types = extract_plotting_info(prompt)
    if plot_detected:
        if st.session_state.selected_column == "All":
            plot_overview(st.session_state.selected_data, plot_types)
        else:
            plot_column_analysis(st.session_state.selected_data, st.session_state.selected_column, plot_types)
