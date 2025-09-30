import os
import requests
import streamlit as st
import time
from secret_api_keys import hugging_face_api_key

# Set the Hugging Face Hub API token as an environment variable
os.environ['HF_TOKEN'] = hugging_face_api_key

# API Configuration
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Available models - you can add more from Hugging Face
MODEL_OPTIONS = [
    "Kwaipilot/KAT-Dev:novita",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-2-70b-chat-hf",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "openchat/openchat-3.5-1210"
]

def query_chat_api(messages, model, max_retries=3, temperature=0.7, max_tokens=500):
    """
    Query the Hugging Face Router API with error handling and retries.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model identifier string
        max_retries: Number of retry attempts
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text or error message
    """
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Extract the message content
                if "choices" in result and len(result["choices"]) > 0:
                    message_content = result["choices"][0]["message"]["content"]
                    return message_content.strip()
                else:
                    return "Error: No response generated"
            
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Rate limit exceeded. Please wait a minute and try again."
            
            elif response.status_code == 401:
                return "Error: Invalid API token. Please check your Hugging Face API key."
            
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    st.warning(f"Model is loading... Retrying in 5 seconds (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                    continue
                else:
                    return "Error: Model is currently unavailable. Try a different model."
            
            else:
                error_msg = response.json().get('error', response.text)
                return f"Error: API returned status {response.status_code} - {error_msg}"
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"Request timed out. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
                continue
            else:
                return "Error: Request timed out after multiple attempts."
        
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.warning(f"Connection error. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
                continue
            else:
                return "Error: Could not connect to Hugging Face API. Check your internet connection."
        
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Unexpected error: {str(e)}. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                return f"Error: {str(e)}"
    
    return "Error: All retry attempts failed."

def generate_titles(topic, model):
    """Generate blog titles using chat API."""
    messages = [
        {
            "role": "system",
            "content": "You are a creative blog title generator. Generate exactly 10 numbered blog titles, one per line."
        },
        {
            "role": "user",
            "content": f"Generate 10 creative, attention-grabbing blog post titles about '{topic}'. Target audience: beginners and tech enthusiasts. Format: numbered list 1-10, no explanations."
        }
    ]
    
    return query_chat_api(messages, model, temperature=0.8, max_tokens=400)

def generate_blog_content(title, keywords, blog_length, model):
    """Generate blog content using chat API."""
    messages = [
        {
            "role": "system",
            "content": f"You are a professional blog writer. Write informative, engaging, and well-structured blog posts of approximately {blog_length} words."
        },
        {
            "role": "user",
            "content": f"""Write a comprehensive blog post with the following specifications:

Title: {title}
Keywords to include: {keywords}
Target length: {blog_length} words
Target audience: beginners

Structure your blog post with:
1. An engaging introduction
2. Well-organized main content with clear points
3. A strong conclusion

Make it informative, easy to understand, and engaging."""
        }
    ]
    
    return query_chat_api(messages, model, temperature=0.7, max_tokens=min(blog_length * 2, 2000))

# Streamlit UI
st.set_page_config(page_title="AI Blog Generator", page_icon="✍️", layout="wide")

st.title("✍️ AI Blog Content Assistant")
st.header("Create High-Quality Blog Content with Advanced AI Models")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose AI Model:",
        MODEL_OPTIONS,
        index=0,
        help="Different models have different strengths. Try different ones for best results."
    )
    
    st.info(f"**Current Model:** {selected_model.split('/')[-1]}")
    
    st.header("📋 How to Use")
    st.write("""
    1. **Generate Titles**: Enter a topic and get 10 creative titles
    2. **Add Keywords**: Add relevant keywords for SEO
    3. **Generate Blog**: Create full blog post with one click
    """)
    
    st.header("💡 Tips")
    st.write("""
    - Be specific with your topic
    - Add 3-5 relevant keywords
    - Start with 300-500 words
    - Try different models for variety
    """)
    
    if st.button("🔄 Clear All Cache"):
        if 'title_results' in st.session_state:
            del st.session_state['title_results']
        if 'blog_results' in st.session_state:
            del st.session_state['blog_results']
        st.success("Cache cleared!")
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

# Title Generation Section
with col1:
    st.subheader("🎯 Step 1: Generate Titles")
    with st.container(border=True):
        topic_name = st.text_input(
            "Enter your blog topic:",
            placeholder="e.g., Machine Learning for Beginners",
            key="topic_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generate_titles_btn = st.button("✨ Generate Titles", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                if 'title_results' in st.session_state:
                    del st.session_state['title_results']
                    st.rerun()
        
        if generate_titles_btn:
            if topic_name.strip():
                with st.spinner('🔄 Generating titles... This may take 30-60 seconds...'):
                    titles = generate_titles(topic_name.strip(), selected_model)
                    st.session_state['title_results'] = titles
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a topic first.")
        
        # Display title results
        if 'title_results' in st.session_state:
            if st.session_state['title_results'].startswith("Error"):
                st.error(st.session_state['title_results'])
            else:
                st.success("✅ Titles generated successfully!")
                st.text_area(
                    "Generated Titles:",
                    value=st.session_state['title_results'],
                    height=300,
                    key="titles_display"
                )

# Blog Generation Section
with col2:
    st.subheader("📝 Step 2: Generate Blog Post")
    with st.container(border=True):
        title_of_the_blog = st.text_input(
            "Blog Title:",
            placeholder="Paste or type your chosen title",
            key="title_input"
        )
        
        num_of_words = st.slider(
            "Target Word Count:",
            min_value=100,
            max_value=1000,
            step=50,
            value=400
        )
        
        # Keyword management
        st.write("**Keywords (SEO):**")
        
        if 'keywords' not in st.session_state:
            st.session_state['keywords'] = []
        
        col_kw1, col_kw2 = st.columns([3, 1])
        with col_kw1:
            keyword_input = st.text_input(
                "Add keyword:",
                placeholder="e.g., Python, AI, Tutorial",
                key="keyword_input",
                label_visibility="collapsed"
            )
        with col_kw2:
            if st.button("➕ Add", use_container_width=True):
                if keyword_input.strip() and keyword_input.strip() not in st.session_state['keywords']:
                    st.session_state['keywords'].append(keyword_input.strip())
                    st.rerun()
        
        # Display current keywords
        if st.session_state['keywords']:
            keywords_html = " ".join([
                f'<span style="background-color:#e8f4f8;padding:5px 10px;margin:3px;border-radius:15px;display:inline-block;">🏷️ {kw}</span>'
                for kw in st.session_state['keywords']
            ])
            st.markdown(keywords_html, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear All Keywords"):
                st.session_state['keywords'] = []
                st.rerun()
        
        st.write("")  # Spacing
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generate_blog_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear Blog", use_container_width=True):
                if 'blog_results' in st.session_state:
                    del st.session_state['blog_results']
                    st.rerun()

if generate_blog_btn:
    if title_of_the_blog.strip():
        formatted_keywords = ', '.join(st.session_state['keywords']) if st.session_state['keywords'] else 'general topics'
        
        with st.spinner('🔄 Generating blog content... This may take 1-2 minutes...'):
            blog_content = generate_blog_content(
                title_of_the_blog.strip(),
                formatted_keywords,
                num_of_words,
                selected_model
            )
            st.session_state['blog_results'] = {
                'title': title_of_the_blog.strip(),
                'content': blog_content,
                'keywords': formatted_keywords
            }
            st.rerun()
    else:
        st.warning("⚠️ Please enter a blog title first.")

# Display blog results in full width
if 'blog_results' in st.session_state:
    st.divider()
    st.subheader("📄 Generated Blog Post")
    
    if st.session_state['blog_results']['content'].startswith("Error"):
        st.error(st.session_state['blog_results']['content'])
    else:
        with st.container(border=True):
            st.markdown(f"### {st.session_state['blog_results']['title']}")
            st.caption(f"Keywords: {st.session_state['blog_results']['keywords']}")
            st.markdown("---")
            st.write(st.session_state['blog_results']['content'])
        
        # Download options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            download_content = f"# {st.session_state['blog_results']['title']}\n\nKeywords: {st.session_state['blog_results']['keywords']}\n\n{st.session_state['blog_results']['content']}"
            st.download_button(
                label="📥 Download as TXT",
                data=download_content,
                file_name=f"{st.session_state['blog_results']['title'].replace(' ', '_')[:50]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            # Markdown version
            markdown_content = f"# {st.session_state['blog_results']['title']}\n\n**Keywords:** {st.session_state['blog_results']['keywords']}\n\n---\n\n{st.session_state['blog_results']['content']}"
            st.download_button(
                label="📥 Download as MD",
                data=markdown_content,
                file_name=f"{st.session_state['blog_results']['title'].replace(' ', '_')[:50]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Word count
        word_count = len(st.session_state['blog_results']['content'].split())
        st.info(f"📊 Word Count: {word_count} words")

# Footer
st.divider()
st.caption("💡 Powered by Hugging Face Chat Completion API | Built with Streamlit")