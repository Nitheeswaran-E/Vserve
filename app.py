import os
import pandas as pd
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

class HRChatbot:
    def __init__(self):

        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_ENGINE"),
            model_name=os.getenv("AZURE_OPENAI_MODEL"),
            temperature=0.0,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        
        self.embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        

        self.load_faq_data()
        

        self.setup_similarity_matching()
        

        self.setup_pandas_agent()
    
    def load_faq_data(self):
        
        self.df = pd.read_excel('C:/Users/Nitheeswaran/Desktop/Nitheesh/vserve/chatbot_hr/FAQ_Dataset.xlsx')
        print(f"Loaded {len(self.df)} FAQ entries")
      
           
    
    def setup_similarity_matching(self):
        """Setup TF-IDF vectorizer for question similarity matching"""
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.question_vectors = self.vectorizer.fit_transform(self.df['Question'].tolist())
    
    def setup_pandas_agent(self):
        """Setup LangChain pandas dataframe agent"""
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True  
        )
    
    def find_best_match(self, user_query, threshold=0.3):
        """Find the best matching FAQ using TF-IDF similarity"""
       
        query_vector = self.vectorizer.transform([user_query])
        
        
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > threshold:
            return {
                'question': self.df.iloc[best_match_idx]['Question'],
                'answer': self.df.iloc[best_match_idx]['Answer'],
                'similarity': best_similarity,
                'match_found': True
            }
        
        return {'match_found': False, 'similarity': best_similarity}
    
    def get_pandas_response(self, user_query):
        """Get response using pandas agent for complex queries"""
        try:
            prompt = f"""
            You are an HR assistant. Use the FAQ dataframe to answer the following question: {user_query}
            
            Guidelines:
            - If the question matches an existing FAQ, return the corresponding answer
            - If you need to search through the data, use pandas operations
            - Provide helpful and professional responses
            - If no relevant information is found, say so politely
            
            Question: {user_query}
            """
            
            response = self.pandas_agent.run(prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error processing your question: {str(e)}"
    
    def get_response(self, user_query):
        """Main method to get chatbot response"""
       
        match_result = self.find_best_match(user_query)
        
        if match_result['match_found']:
            return {
                'response': match_result['answer'],
                'method': 'similarity_match',
                'confidence': match_result['similarity'],
                'matched_question': match_result['question']
            }
        
      
        pandas_response = self.get_pandas_response(user_query)
        
        return {
            'response': pandas_response,
            'method': 'pandas_agent',
            'confidence': 0.5  
        }


chatbot = HRChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a message'}), 400
        
       
        result = chatbot.get_response(user_message)
        
        return jsonify({
            'response': result['response'],
            'method': result['method'],
            'confidence': result.get('confidence', 0),
            'matched_question': result.get('matched_question', '')
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/faqs')
def show_faqs():
    """Endpoint to show all available FAQs"""
    faqs = chatbot.df.to_dict('records')
    return jsonify(faqs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)