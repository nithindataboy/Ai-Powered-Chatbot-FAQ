import os
import torch
import streamlit as st
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from dotenv import load_dotenv
from fuzzywuzzy import process
import google.generativeai as genai  
import base64

st.set_page_config(page_title="AI College FAQ Chatbot", page_icon="🎓", layout="centered")
#adding img
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .text-container {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    h1 {{
        color: #2C3E50;
        font-family: 'Arial', sans-serif;
        font-size: 48px; /* Increased font size */
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); /* Shadow for better visibility */
        margin-bottom: 20px;
    }}
    h2 {{
        color: #2980B9;
        font-family: 'Arial', sans-serif;
        font-size: 30px;
        margin-top: 20px;
    }}
    h3 {{
        color: #34495E;
        font-family: 'Arial', sans-serif;
        font-size: 26px;
    }}
    p {{
        color: #555;
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        line-height: 1.6;
    }}
    .stTextInput {{
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #2980B9;
        border-radius: 5px;
        padding: 10px;
        font-size: 18px;
    }}
    .stButton {{
        background-color: #2980B9; /* Blue */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s;
        font-size: 18px;
        font-weight: bold;
    }}
    .stButton:hover {{
        background-color: #3498DB;
        transform: scale(1.05);
    }}
    .card {{
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }}
    .suggestion-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .faq-section {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .voice-response-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    /* AI Search Button */
        .ai-search-button {{
            background-color: #0073e6 !important; /* Deep Blue */
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        }}
        .ai-search-button:hover {{
            background-color: #005bb5 !important; /* Darker Blue */
            transform: scale(1.05);
        }}

        /* Google CSI Search Button */
        .google-search-button {{
            background-color: #2E8B57 !important; /* Professional Dark Green */
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        }}
        .google-search-button:hover {{
            background-color: #206040 !important; /* Darker Green */
            transform: scale(1.05);
        }}

        .card {{
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
# Call the function to set the background
set_background(r"C:\hits student assistance faq chatbot\1.jpeg")
# ✅ Load API keys
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  

# ✅ Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")  

# ✅ College FAQ dataset (RAG Source for BERT)
qa_pairs = {
    "college name" : "Holy Mary Institute of technolgy and science hits",
    "About":"Established in 2001, The college strives to impart qualitative technical education through innovative teaching methodologies. The college strives to establish itself as a world class education center for teaching, learning, research & training, with modern facilities and vast expanded landscape. The college campus provides perfect ambience for higher learning. HITS is an inspiring institute with the sate of the art facilities for students to meet the requirements of the industry.",
    "Location":"Holy Mary Institute of Technology & Science (HITS) is located at Bogaram (V), Keesara (M), R.R. District, Keesara, Medchal, Telangana, India, with the postal code 501301. The institute was established in 2001 and is affiliated with Jawaharlal Nehru Technological University, Hyderabad (JNTUH). HITS offers a range of undergraduate and postgraduate programs in engineering, technology, and management",
    "contact us":"Phone: 08415-200255/466, Email: examshitscoe@hmgi.ac.in ,directoradmissions@hmgi.ac.in, Mobile: 9848077751",
    "official website":"​The official website of Holy Mary Institute of Technology & Science (HITS) is https://www.hits.ac.in/. ",
    "admission process": "Admissions at HITS are through **EAMCET, ECET, and Management Quota**. The process involves entrance exam scores, document verification, and counseling.",
    "chairman of HITS": "The chairman of HITS is **Dr. Siddarth Reddy Arminanda**, known for his contributions to education and institutional development.",
    "departments available": "HITS offers **CSE, AI/ML, IoT, Mechanical, Civil, ECE, etc.** with modern labs, research centers, and highly qualified faculty.",
    "placements": "Top recruiters include **TCS, Infosys, Wipro, Cognizant, and Capgemini**. The college provides placement training, internships, and career guidance.",
    "documents for admission": "Required documents: **10th & 12th marksheets, TC, Entrance Score, Aadhar, Passport-size photos, and Category Certificate (if applicable).**",
   
    "courses offered": "HITS offers a variety of programs including **B.Tech** in fields like Artificial Intelligence, Civil Engineering, Computer Science, and more. **M.Tech** programs and **MBA** specializations are also available.",
    "eligibility criteria": "Eligibility varies by program. For **B.Tech**, candidates must have completed 10+2 with relevant subjects and qualifying entrance exams. Detailed criteria are available on the official website.",
    "fee structure": "The fee structure differs across programs and categories. For the most accurate and up-to-date information, please refer to the **Fee Structure** section on the HITS website.",
    "scholarships available": "HITS provides various scholarships based on merit and other criteria. Prospective students are encouraged to check the **Scholarships** section on the website for detailed information.",
    "admission deadlines": "Admission deadlines vary annually. It's advisable to consult the **Admissions** section of the HITS website or contact the admissions office directly for the current academic year's deadlines.",
    "placement opportunities": "HITS has a dedicated **Career Development Center** that facilitates placements with top recruiters such as TCS, Infosys, and Wipro. The center also offers training and internship opportunities.",
    "hostel facilities": "Separate hostel facilities are available for both boys and girls, equipped with necessary amenities to ensure a comfortable stay for students.",
    "transportation services": "HITS offers transportation services across various routes in Hyderabad, ensuring safe and convenient travel for students and staff.",
    "campus infrastructure": "The campus boasts modern infrastructure, including well-equipped labs, libraries, sports facilities, and classrooms designed to enhance the learning experience.",
    "international collaborations": "HITS has established tie-ups with international universities, providing students with opportunities for exchange programs and global exposure.",
    "anti-ragging policy": "HITS maintains a strict **Anti-Ragging & Disciplinary Committee** to ensure a safe and conducive environment for all students.",
    "grievance redressal": "Students can address their concerns through the **Online Grievance Redressal** system available on the HITS website, ensuring timely resolution of issues.",
    "research programs": "The institute encourages research through various **Research Programmes & Workshops**, fostering innovation and academic growth among students and faculty.",
    "industry tie-ups": "HITS has collaborations with industries like **Microsoft**, **IBM**, and **Oracle**, providing students with exposure to current technologies and industry practices.",
    "extracurricular activities": "A range of extracurricular activities, including sports, cultural events, and technical clubs, are available to support the holistic development of students."


}

# ✅ Load AI models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Speech Engine (Fix: Added Stop Button)
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# 🔥 **Google CSI Search (With Images & 2-5 Line Answers)**
def google_search(query, api_key, cse_id):
    if not api_key or not cse_id:
        return "❌ Error: API keys are missing."
    
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query + " site:hits.ac.in", cx=cse_id, num=5).execute()
        results = res.get("items", [])
        
        final_results = []
        for result in results:
            title = result.get("title", "No Title")
            link = result.get("link", "#")
            snippet = result.get("snippet", "No additional details available.")

            # ✅ Ensure 2-5 line summary
            if len(snippet.split()) < 15:
                snippet += " This information is related to HITS. Visit the official website for full details."

            final_results.append({"title": title, "link": link, "snippet": snippet, "image": None})

            # ✅ Add Image if available
            if 'pagemap' in result and 'cse_image' in result['pagemap']:
                for img in result['pagemap']['cse_image']:
                    final_results[-1]['image'] = img['src']
                    break  # Use only the first available image

        return final_results
    except Exception as e:
        return f"Error: {str(e)}"

# 🔥 **Find Best Match for Short Queries**
def find_best_match(query, documents):
    best_match, score = process.extractOne(query, documents.keys())
    return best_match if score > 70 else None  

# 🔥 **Semantic Search for Best Context**
def semantic_search(query, documents):
    best_match = find_best_match(query, documents)
    return documents[best_match] if best_match else None  

# 🔥 **BERT Model for Contextual Answers (Fix RAG Accuracy)**
def get_bert_answer(question, context):
    inputs = tokenizer.encode_plus(
        question, context, add_special_tokens=True, return_tensors="pt",
        padding=True, truncation=True, max_length=512
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    if start_index >= end_index:
        return None  

    answer_tokens = input_ids[0][start_index:end_index+1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# 🔥 **Convert AI Answer to Speech (Added Stop Button)**
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def stop_speech():
    engine.stop()

# 🔥 **Gemini AI for Additional Suggestions (Now Fully Fixed)**
def get_gemini_suggestion(query):
    try:
        response = gemini_model.generate_content(f"Provide a relevant, helpful, and specific suggestion related to: {query}")
        return response.text.strip() if response else "No suggestion available."
    except Exception as e:
        return f"Error: {str(e)}"

# 🔥 **Streamlit Chatbot**

def run_chatbot():
    st.markdown("""
    <style>
    .title-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .faq-section {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .voice-response-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Title Section
    st.markdown('<div class="title-container"><h1>🎓 AI-Powered College FAQ Chatbot</h1></div>', unsafe_allow_html=True)

    # Text Input Area
    user_query = st.text_input("🔍 Ask a question about the college:", placeholder="Type your question here...")
    

    # Voice Response Section
    st.markdown(
    """
    <style>
        .voice-container {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
            width: fit-content;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Use a div container for the checkbox
    st.markdown('<div class="voice-container">🔊 Enable Voice Response</div>', unsafe_allow_html=True)

# Add the actual checkbox
    speak_mode = st.checkbox("")

    
    col1, col2 = st.columns(2)

    # FAQ Section
    st.markdown(
    """
    <style>
        .selectbox-container {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            display: inline-block;
        }
        .selectbox-container label {
            font-weight: bold;
            font-size: 16px;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown('<div class="faq-section"><h2>📌 Frequently Asked Questions</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="selectbox-container">Select a question:</div>', unsafe_allow_html=True)
    selected_question = st.selectbox("", ["-- Select a question --"] + list(qa_pairs.keys()))
    

    if selected_question != "-- Select a question --":
        answer = qa_pairs[selected_question]  # Directly get the answer from the dataset
        st.markdown(f"<div class='text-container'><h3>🧠 AI Answer:</h3><p>{answer}</p></div>", unsafe_allow_html=True)
        
        if speak_mode:
            speak_text(answer)

    # AI Search
    if col1.button("🔍 AI Search"):
        context = semantic_search(user_query, qa_pairs)
        if context:
            answer = get_bert_answer(user_query, context) or context
            st.markdown(f"<div class='text-container'><h3>🧠 AI Answer:</h3><p>{answer}</p></div>", unsafe_allow_html=True)
            if speak_mode:
                speak_text(answer)
        else:
            st.write("❌ AI couldn't find an answer. Try rephrasing the question.")

    # Google CSI Search
    if col2.button("🌍 Google CSI Search"):
        search_results = google_search(user_query, API_KEY, CSE_ID)

        if isinstance(search_results, str):
            st.write(search_results)
        elif search_results:
            for result in search_results:
                st.markdown(f"""
                <div class="card">
                    <h4>{result['title']}</h4>
                    <p>{result['snippet']}</p>
                    <a href="{result['link']}" target="_blank">Read More</a>
                </div>
                """, unsafe_allow_html=True)
                if result['image']:
                    st.image(result['image'], width=300)  # Adjust width as needed

            # Gemini AI Suggestion Under Google Results
            suggestion = get_gemini_suggestion(user_query)
            if suggestion:
                st.markdown(f"""
                <div class="suggestion-container">
                    <h3>✨ AI Suggestion:</h3>
                    <p>{suggestion}</p>
                </div>
                """, unsafe_allow_html=True)
                if speak_mode:
                    speak_text(suggestion)


    # Stop Speech Button
    if st.button("⏹️ Stop Voice"):
        stop_speech()
        st.write("🔇 Voice Stopped.")

# Run the Chatbot
if __name__ == "__main__":
    run_chatbot()