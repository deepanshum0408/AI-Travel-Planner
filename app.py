def initialize_agent():
    if 'agent' not in st.session_state:
        from agents.agent import Agent
        st.session_state.agent = Agent()
def process_query(user_input):
    if user_input:
        try:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}}

            print(f"process_query: user_input = {user_input}")
            print(f"process_query: messages = {messages}")
            print(f"process_query: config = {config}")

            result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)

            print(f"process_query: result = {result}")

            st.subheader('Travel Information')
            st.markdown(result['messages'][-1].content, unsafe_allow_html=True)

            st.session_state.travel_info = result['messages'][-1].content

        except Exception as e:
            print(f"process_query: error = {e}")
            st.error(f'Error: {e}')
    else:
        print("process_query: No user input provided.")
        st.error('Please enter a travel query.')
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition


# Load environment variables
load_dotenv()

# Initialize Groq LLM
def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        st.stop()
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.3
    )

def send_email(sender_email, receiver_email, subject, travel_info):
    """
    Send itinerary email using SendGrid API.
    You must set SENDGRID_API_KEY as an environment variable,
    and the 'sender_email' must be verified in your SendGrid account.
    """
    try:
        # Check if API key exists
        api_key = os.environ.get("SENDGRID_API_KEY")
        if not api_key:
            st.error("❌ SENDGRID_API_KEY not found in environment variables. Please add it to your .env file.")
            return
        
        # Check if API key looks valid (starts with SG.)
        if not api_key.startswith("SG."):
            st.error("❌ Invalid SendGrid API key format. Should start with 'SG.'")
            return

        # Resolve sender email: prefer provided value, otherwise use FROM_EMAIL env
        resolved_sender = sender_email or os.environ.get("FROM_EMAIL")
        if not resolved_sender:
            st.error("❌ Missing sender email. Set FROM_EMAIL in your .env or provide a sender.")
            return
            
        # Generate PDF attachment from itinerary content
        html_for_pdf = _markdown_to_html(travel_info)
        pdf_bytes = _generate_pdf_from_html(html_for_pdf)
        if not pdf_bytes:
            st.error("❌ Failed to generate PDF. Please try again.")
            return

        import base64
        encoded = base64.b64encode(pdf_bytes).decode()

        message = Mail(
            from_email=resolved_sender,
            to_emails=receiver_email,
            subject=subject,
            plain_text_content=" "  # minimal content; no body besides attachment
        )

        attachment = Attachment(
            FileContent(encoded),
            FileName("itinerary.pdf"),
            FileType("application/pdf"),
            Disposition("attachment"),
        )
        message.add_attachment(attachment)
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        if response.status_code == 202:
            st.success(f"📧 Email sent successfully!\n\nFrom: {resolved_sender}\nTo: {receiver_email}\nSubject: {subject}")
        elif response.status_code == 403:
            st.error("❌ 403 Forbidden: Check that your sender email is verified in SendGrid and your API key has 'Mail Send' permissions.")
        else:
            st.error(f"❌ Failed to send email (status {response.status_code}). Check SendGrid dashboard for details.")
        
        # Clear session state
        for key in ['travel_info', 'thread_id']:
            st.session_state.pop(key, None)
    except Exception as e:
        st.error(f'❌ Error sending email: {e}')
        st.error("💡 Common fixes: 1) Verify sender email in SendGrid, 2) Check API key permissions, 3) Ensure SENDGRID_API_KEY is in .env file")

def render_custom_css():
    st.markdown(
        '''
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        /* Main App Background - Modern Gradient */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
            background-size: 400% 400%;
            animation: gradientShift 20s ease infinite;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main Container */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Expand all content elements to full width */
        .main .element-container,
        .main .stMarkdown,
        .main [data-testid="stMarkdownContainer"],
        .main [data-baseweb="base-input"] {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Hero Image Section */
        .hero-image-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto 2.5rem;
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            position: relative;
            transition: transform 0.3s ease;
        }
        
        .hero-image-container:hover {
            transform: translateY(-5px);
        }
        
        .hero-image {
            width: 100%;
            height: 350px;
            object-fit: cover;
            display: block;
        }
        
        /* Title Styling */
        .main-title {
            font-size: 3.2em;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0.8rem;
            font-weight: 800;
            font-family: 'Poppins', sans-serif;
            letter-spacing: -1px;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            animation: fadeInDown 0.8s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .sub-title {
            font-size: 1.3em;
            color: rgba(255, 255, 255, 0.95);
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
            font-family: 'Inter', sans-serif;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            animation: fadeInUp 0.8s ease 0.2s both;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Center Container */
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            padding: 1rem 0;
        }
        
        /* Image Gallery */
        .image-gallery {
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            margin: 2rem 0 3rem;
            flex-wrap: wrap;
        }
        
        .gallery-image {
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            overflow: hidden;
        }
        
        .gallery-image:hover {
            transform: translateY(-10px) scale(1.03);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }
        
        /* Query Container - No Box */
        .query-container {
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            background: none;
            padding: 0;
            border: none;
            box-shadow: none;
            backdrop-filter: none;
            border-radius: 0;
        }
        
        /* Text Area Styling - Enhanced */
        .stTextArea textarea {
            border-radius: 20px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            padding: 2rem;
            font-size: 1.2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            background: transparent;
            color: #ffffff;
            line-height: 1.7;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.15), 
                        0 4px 12px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
            min-height: 140px;
        }
        
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.7);
            font-weight: 400;
            opacity: 0.9;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea;
            background: transparent;
            box-shadow: 0 0 0 6px rgba(102, 126, 234, 0.2),
                        0 15px 50px rgba(102, 126, 234, 0.25),
                        0 6px 20px rgba(0, 0, 0, 0.1);
            outline: none;
            transform: translateY(-2px);
        }
        
        .stTextArea textarea:hover:not(:focus) {
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 12px 45px rgba(102, 126, 234, 0.2),
                        0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Button Styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 1rem 2rem;
            font-size: 1.2rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            margin-top: 1.5rem;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            text-transform: none;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Results Container */
        .results-container {
            background: transparent;
            backdrop-filter: none;
            border-radius: 0;
            padding: 1.5rem 0;
            margin-top: 1.5rem;
            border: none;
            box-shadow: none;
            width: 100%;
            max-width: 100%;
        }
        
        .results-container .stMarkdown,
        .results-container .stMarkdown p,
        .results-container .stMarkdown li,
        .results-container .stMarkdown div,
        .results-container .stMarkdown h1,
        .results-container .stMarkdown h2,
        .results-container .stMarkdown h3,
        .results-container .stMarkdown strong,
        .results-container .stMarkdown em {
            color: #ffffff !important;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
        }
        
        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 0.9rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(30, 41, 59, 0.3);
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(30, 41, 59, 0.4);
        }
        
        /* Form Elements */
        .stTextInput input {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 0.9rem 1.2rem;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            background: #f8fafc;
            color: #000000;
        }
        
        .stTextInput input:focus {
            border-color: #667eea;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            color: #000000;
        }
        
        .stTextInput label {
            color: #000000 !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
        }
        
        .stRadio label {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: #000000 !important;
            font-size: 1rem;
        }
        
        /* Form container text colors */
        .stForm label,
        .stForm .stTextInput label,
        .stForm .stRadio label {
            color: #000000 !important;
        }
        
        .stForm input[type="text"],
        .stForm input[type="email"] {
            color: #000000 !important;
        }
        
        /* Form Submit Button */
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 0.9rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            width: 100%;
        }
        
        .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        }
        
        /* Success/Error Messages */
        .stSuccess {
            border-radius: 12px;
            border-left: 4px solid #10b981;
            padding: 1rem;
            background: rgba(16, 185, 129, 0.1);
        }
        
        .stError {
            border-radius: 12px;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
        }
        
        .stWarning {
            border-radius: 12px;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            background: rgba(245, 158, 11, 0.1);
        }
        
        .stInfo {
            border-radius: 12px;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            background: rgba(59, 130, 246, 0.1);
        }
        
        /* Sidebar Styling - Distinct but Matching Design */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.85) 0%, rgba(118, 75, 162, 0.9) 100%) !important;
            backdrop-filter: blur(20px);
            border-right: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
        }
        
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p {
            color: #ffffff !important;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            font-family: 'Inter', sans-serif;
        }
        
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.4);
            margin: 1.5rem 0;
            border-width: 1px;
        }
        
        .sidebar-image {
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
            padding: 4px;
        }
        
        .sidebar-image:hover {
            transform: translateY(-5px) scale(1.03);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
        }
        
        /* Sidebar tip box styling */
        [data-testid="stSidebar"] div[style*="background"] {
            background: rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255, 255, 255, 0.4) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        
        [data-testid="stSidebar"] div[style*="background"] p,
        [data-testid="stSidebar"] div[style*="background"] strong {
            color: #ffffff !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        }
        
        /* Sidebar scrollbar styling to match */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 8px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        /* Disable Image Fullscreen/Expand */
        .stImage img {
            pointer-events: none !important;
            cursor: default !important;
        }
        
        .stImage > div {
            pointer-events: none !important;
        }
        
        .stImage > div > img {
            pointer-events: none !important;
            cursor: default !important;
        }
        
        /* Remove expand button/overlay */
        .stImage button,
        .stImage [data-testid="stImage"] button,
        .stImage .image-overlay {
            display: none !important;
        }
        
        /* Prevent image click expansion */
        div[data-testid="stImage"] {
            pointer-events: none !important;
        }
        
        div[data-testid="stImage"] img {
            pointer-events: none !important;
            cursor: default !important;
        }
        
        /* Markdown Content */
        .stMarkdown {
            font-family: 'Inter', sans-serif;
            line-height: 1.7;
            color: #ffffff;
            font-size: 1.05rem;
            max-width: 100%;
            width: 100%;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .stMarkdown p, .stMarkdown li, .stMarkdown div, .stMarkdown strong, .stMarkdown em {
            color: #ffffff;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Style the header to match the design */
        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Style the menu button */
        button[data-testid="baseButton-header"] {
            color: #ffffff !important;
        }
        
        /* Style the deploy button */
        [data-testid="stDeployButton"] {
            color: #ffffff !important;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.2em;
            }
            .sub-title {
                font-size: 1.1em;
            }
            .query-container {
                padding: 2rem 1.5rem;
            }
            .hero-image-container {
                margin-bottom: 1.5rem;
            }
        }
        </style>
        ''', unsafe_allow_html=True)

def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    
    # Hero Image Section
    try:
        st.markdown('<div class="hero-image-container">', unsafe_allow_html=True)
        st.image('images/a.jpg', use_container_width=True, caption='')
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        pass
    
    # Title Section
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Plan your perfect trip with AI-powered flight and hotel recommendations</div>', unsafe_allow_html=True)
    
    # Image Gallery with better spacing
    try:
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image('images/b.jpg', use_container_width=True, caption='')
        with col2:
            st.image('images/c.jpg', use_container_width=True, caption='')
        with col3:
            st.image('images/ai-travel.png', use_container_width=True, caption='')
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        pass
    
    # Query Container with improved styling
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="font-family: \'Poppins\', sans-serif; color: #1e293b; margin-bottom: 1.5rem; text-align: center; font-size: 1.5rem;">Tell us about your dream destination</h3>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        'Travel Query',
        height=100,
        key='query',
        placeholder='Enter your travel query and get flight and hotel information:',
        label_visibility='collapsed'
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Sidebar - Integrated Design
    with st.sidebar:
        st.markdown("""
        <h3 style="font-family: 'Poppins', sans-serif; color: #ffffff; text-align: center; margin-bottom: 1.5rem; font-size: 2.0rem; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);">
            🌟 Travel Inspiration
        </h3>
        """, unsafe_allow_html=True)
        st.markdown("---")
        try:
            st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
            st.image('images/b.jpg', caption='', use_container_width=True, output_format='auto')
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
            st.image('images/c.jpg', caption='', use_container_width=True, output_format='auto')
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
            st.image('images/ai-travel.png', caption='', use_container_width=True, output_format='auto')
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass
        
        st.markdown("---")
        st.markdown("""
        <div style="padding: 1.2rem; background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px); border-radius: 16px; margin-top: 1.5rem; border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);">
            <p style="font-size: 0.95rem; color: #ffffff; margin: 0; line-height: 1.6; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                <strong style="color: #ffffff; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">💡 Tip:</strong> Include departure city, destination, dates, and hotel preferences for best results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    return user_input

def process_query(user_input):
    if user_input:
        try:
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}}

            print(f"process_query: user_input = {user_input}")
            print(f"process_query: messages = {messages}")
            print(f"process_query: config = {config}")

            with st.spinner('🔍 Searching for flights and hotels...'):
                result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)

            print(f"process_query: result = {result}")

            # Display results in a styled container
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown('<h2 style="font-family: \'Poppins\', sans-serif; color: #ffffff; margin-bottom: 1.5rem; text-align: center; font-size: 2rem; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);">✨ Your Travel Itinerary</h2>', unsafe_allow_html=True)
            st.markdown(result['messages'][-1].content, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state.travel_info = result['messages'][-1].content

        except Exception as e:
            print(f"process_query: error = {e}")
            st.error(f'❌ Error processing your request: {e}')
            st.info('💡 Please try again with a different query or check your internet connection.')
    else:
        print("process_query: No user input provided.")
        st.warning('⚠️ Please enter a travel query to get started.')

def _generate_pdf_from_html(html_content: str):
    """
    Generate PDF bytes from HTML using Playwright (Chromium).
    Returns bytes on success, or None on failure with a user-facing error.
    """
    try:
        # Fix Windows asyncio policy for subprocess used by Playwright
        import sys
        import asyncio
        if sys.platform.startswith('win'):
            try:
                # Proactor policy supports subprocess on Windows
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            except Exception:
                pass
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        st.error('Playwright is not installed. Run: pip install playwright && playwright install chromium')
        return None

    try:
        with sync_playwright() as p:
            # Ensure headless (required for page.pdf)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Improve rendering reliability
            page.set_viewport_size({"width": 1024, "height": 1280})
            page.set_content(html_content, wait_until="load")
            page.emulate_media(media="print")
            pdf_bytes = page.pdf(format="A4", print_background=True, margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"})
            browser.close()
            return pdf_bytes
    except Exception as e:
        st.error(f'Error generating PDF: {e!r}')
        st.info('If this persists: 1) Ensure "playwright install chromium" completed successfully, 2) Restart the app, 3) Try running the app from a terminal with permissions.')
        return None

def _markdown_to_html(content: str) -> str:
    """Convert mixed Markdown/HTML to full HTML with basic styles for PDF."""
    html_body = None
    try:
        import markdown  # type: ignore
        # Convert markdown to HTML while preserving any inline HTML
        html_body = markdown.markdown(
            content,
            extensions=[
                'extra',
                'sane_lists',
                'nl2br',
                'smarty'
            ]
        )
    except Exception:
        # Fallback: minimal escaping and line breaks
        import html as _html
        html_body = _html.escape(content).replace('\n', '<br>')

    # Basic CSS for readability in PDF
    styles = """
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color: #222; line-height: 1.5; }
      h1, h2, h3 { margin: 0.6rem 0 0.4rem; font-weight: 700; }
      h1 { font-size: 1.6rem; }
      h2 { font-size: 1.3rem; }
      h3 { font-size: 1.1rem; }
      p { margin: 0.3rem 0; }
      ul, ol { margin: 0.2rem 0 0.6rem 1.2rem; }
      li { margin: 0.15rem 0; }
      hr { border: none; border-top: 1px solid #ddd; margin: 0.8rem 0; }
      img { max-width: 140px; height: auto; display: inline-block; margin: 0.2rem 0; }
      .section-title { margin-top: 0.8rem; font-size: 1.2rem; font-weight: 700; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 6px; background: #f2f4f7; font-size: 0.85rem; }
      .muted { color: #666; }
      a { color: #0b63c9; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .block { margin: 0.6rem 0; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """

    # Wrap in a clean HTML document
    return f"""
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        {styles}
      </head>
      <body>
        {html_body}
      </body>
    </html>
    """

def render_pdf_download():
    if 'travel_info' not in st.session_state:
        return
    html = st.session_state.travel_info
    # Convert markdown/HTML mix to clean HTML for PDF
    html_for_pdf = _markdown_to_html(html)
    pdf_bytes = _generate_pdf_from_html(html_for_pdf)
    if pdf_bytes:
        st.download_button(
            label='⬇️ Download Itinerary as PDF',
            data=pdf_bytes,
            file_name='itinerary.pdf',
            mime='application/pdf'
        )

def render_email_form():
    send_email_option = st.radio('Do you want to send this information via email?', ('No', 'Yes'))
    if send_email_option == 'Yes':
        with st.form(key='email_form'):
            receiver_email = st.text_input('Your Email (recipient)')
            subject = st.text_input('Email Subject', 'Travel Information')
            submit_button = st.form_submit_button(label='Send Email')

        if submit_button:
            if receiver_email and subject:
                # Sender is optional; pull from FROM_EMAIL env if not provided
                send_email(sender_email=None, receiver_email=receiver_email, subject=subject, travel_info=st.session_state.travel_info)
            else:
                st.error('Please enter your email and subject.')

def main():
    initialize_agent()
    render_custom_css()
    user_input = render_ui()

    # Search button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔍 Search Flights & Hotels', use_container_width=True, type='primary'):
            process_query(user_input)

    if 'travel_info' in st.session_state:
        st.markdown('<div style="margin-top: 3rem;">', unsafe_allow_html=True)
        render_pdf_download()
        render_email_form()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
