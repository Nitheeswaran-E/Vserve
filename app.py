import os
import pandas as pd
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

class HRChatbot:
    def __init__(self):
        print("Initializing HR Chatbot with open source models...")
 
        self.setup_language_model()
        self.setup_embeddings_model()
   
        self.load_faq_data()
      
        self.setup_similarity_matching()
        self.setup_semantic_similarity()
    
    def setup_language_model(self):
        """Setup open source language model for text generation"""
        try:
          
            model_name = "microsoft/DialoGPT-medium"  
            
            
            print(f"Loading language model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
           
            self.text_generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("Language model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading language model: {e}")
          
            self.text_generator = None
    
    def setup_embeddings_model(self):
        """Setup sentence transformer model for semantic similarity"""
        try:
            
            model_name = "all-MiniLM-L6-v2"  # Fast and efficient
           
            
            print(f"Loading embeddings model: {model_name}")
            self.sentence_model = SentenceTransformer(model_name)
            print("Embeddings model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading embeddings model: {e}")
            self.sentence_model = None
    
    def load_faq_data(self):
        """Load FAQ data from Excel file"""
        try:
           
            self.df = pd.read_excel('FAQ_Dataset.xlsx')
            print(f"Loaded {len(self.df)} FAQ entries")
        except Exception as e:
            print(f"Error loading FAQ data: {e}")
            
            self.df = pd.DataFrame({
                'Question': [
                    'What are the company working hours?',
                    'How do I apply for leave?',
                    'What is the dress code policy?',
                    'How do I access my payslip?',
                    'What are the benefits provided?'
                ],
                'Answer': [
                    'Company working hours are 9 AM to 6 PM, Monday to Friday.',
                    'You can apply for leave through the HR portal or by filling out the leave application form.',
                    'The dress code is business casual. Please refer to the employee handbook for details.',
                    'You can access your payslip through the employee self-service portal.',
                    'Benefits include health insurance, retirement plans, and paid time off. Contact HR for complete details.'
                ]
            })
            print(f"Using sample FAQ data with {len(self.df)} entries")
    
    def setup_similarity_matching(self):
        """Setup TF-IDF vectorizer for question similarity matching"""
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.question_vectors = self.vectorizer.fit_transform(self.df['Question'].tolist())
    
    def setup_semantic_similarity(self):
        """Setup semantic similarity using sentence transformers"""
        if self.sentence_model:
            try:
                self.question_embeddings = self.sentence_model.encode(self.df['Question'].tolist())
                print("Question embeddings created successfully!")
            except Exception as e:
                print(f"Error creating question embeddings: {e}")
                self.question_embeddings = None
        else:
            self.question_embeddings = None
    
    def find_best_match_tfidf(self, user_query, threshold=0.3):
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
                'match_found': True,
                'method': 'tfidf'
            }
        
        return {'match_found': False, 'similarity': best_similarity}
    
    def find_best_match_semantic(self, user_query, threshold=0.7):
        """Find the best matching FAQ using semantic similarity"""
        if not self.sentence_model or self.question_embeddings is None:
            return {'match_found': False, 'similarity': 0}
        
        try:
            query_embedding = self.sentence_model.encode([user_query])
            similarities = cosine_similarity(query_embedding, self.question_embeddings).flatten()
            
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > threshold:
                return {
                    'question': self.df.iloc[best_match_idx]['Question'],
                    'answer': self.df.iloc[best_match_idx]['Answer'],
                    'similarity': best_similarity,
                    'match_found': True,
                    'method': 'semantic'
                }
            
            return {'match_found': False, 'similarity': best_similarity}
        
        except Exception as e:
            print(f"Error in semantic matching: {e}")
            return {'match_found': False, 'similarity': 0}
    
    def generate_contextual_response(self, user_query, context_info=None):
        """Generate response using the language model"""
        if not self.text_generator:
            return "I apologize, but I'm unable to process your question at the moment. Please try rephrasing your question or contact HR directly."
        
        try:
           
            if context_info:
                prompt = f"HR Assistant: Based on our FAQ database, here's what I found regarding '{user_query}': {context_info}. Let me provide you with a helpful response:"
            else:
                prompt = f"HR Assistant: I'll help you with your question about '{user_query}'. Here's my response:"
            
            # Generate response
            generated = self.text_generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response = generated[0]['generated_text']
           
            if len(response) > len(prompt):
                response = response[len(prompt):].strip()
            else:
                response = "I understand your question, but I'd recommend checking with HR for the most accurate information."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue generating a response. Please contact HR for assistance."
    
    def search_dataframe(self, user_query):
        """Search through the dataframe for relevant information"""
        query_lower = user_query.lower()
        
        
        question_matches = self.df[self.df['Question'].str.lower().str.contains(query_lower, na=False)]
        answer_matches = self.df[self.df['Answer'].str.lower().str.contains(query_lower, na=False)]
        
      
        all_matches = pd.concat([question_matches, answer_matches]).drop_duplicates()
        
        if not all_matches.empty:
          
            best_match = all_matches.iloc[0]
            return {
                'question': best_match['Question'],
                'answer': best_match['Answer'],
                'match_found': True,
                'method': 'keyword_search'
            }
        
        return {'match_found': False}
    
    def get_response(self, user_query):
        """Main method to get chatbot response"""
      
        semantic_result = self.find_best_match_semantic(user_query, threshold=0.7)
        if semantic_result['match_found']:
            return {
                'response': semantic_result['answer'],
                'method': 'semantic_similarity',
                'confidence': float(semantic_result['similarity']),  # Convert to Python float
                'matched_question': semantic_result['question']
            }
        
        
        tfidf_result = self.find_best_match_tfidf(user_query, threshold=0.3)
        if tfidf_result['match_found']:
            return {
                'response': tfidf_result['answer'],
                'method': 'tfidf_similarity',
                'confidence': float(tfidf_result['similarity']),  # Convert to Python float
                'matched_question': tfidf_result['question']
            }
        
        
        keyword_result = self.search_dataframe(user_query)
        if keyword_result['match_found']:
            return {
                'response': keyword_result['answer'],
                'method': 'keyword_search',
                'confidence': 0.6,
                'matched_question': keyword_result['question']
            }
        
       
        generated_response = self.generate_contextual_response(
            user_query, 
            "I couldn't find a direct match in our FAQ database"
        )
        
        return {
            'response': generated_response,
            'method': 'generated_response',
            'confidence': 0.4
        }



print("Starting HR Chatbot initialization...")
chatbot = HRChatbot()
print("HR Chatbot ready!")

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
            'confidence': float(result.get('confidence', 0)), 
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
