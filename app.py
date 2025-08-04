# app.py - Main Flask API for Railway deployment
from flask import Flask, request, jsonify
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Initialize AI services
pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/draft-personal-email', methods=['POST'])
def draft_personal_email():
    """
    Draft email using Jordan's personal style
    """
    try:
        data = request.json
        incoming_subject = data.get('subject', '')
        incoming_body = data.get('body', '')
        sender_email = data.get('from', '')
        
        incoming_text = f"Subject: {incoming_subject}\n\nFrom: {sender_email}\n\nBody: {incoming_body}"
        
        # Search Jordan's personal database
        index = pinecone_client.Index("jordan-personal-emails")
        similar_responses = find_similar_responses(incoming_text, index, top_k=5)
        
        # Generate personal-style response
        draft_response = generate_personal_response(incoming_text, similar_responses)
        
        return jsonify({
            'success': True,
            'style': 'personal',
            'draft_subject': generate_subject_line(incoming_subject),
            'draft_body': draft_response,
            'confidence_score': calculate_confidence(similar_responses),
            'similar_examples_used': len(similar_responses)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/draft-admin-email', methods=['POST'])
def draft_admin_email():
    """
    Draft email using admin team style
    """
    try:
        data = request.json
        incoming_subject = data.get('subject', '')
        incoming_body = data.get('body', '')
        sender_email = data.get('from', '')
        
        incoming_text = f"Subject: {incoming_subject}\n\nFrom: {sender_email}\n\nBody: {incoming_body}"
        
        # Search admin team database
        index = pinecone_client.Index("admin-team-emails")
        similar_responses = find_similar_responses(incoming_text, index, top_k=5)
        
        # Generate admin-style response
        draft_response = generate_admin_response(incoming_text, similar_responses)
        
        return jsonify({
            'success': True,
            'style': 'admin',
            'draft_subject': generate_subject_line(incoming_subject),
            'draft_body': draft_response,
            'confidence_score': calculate_confidence(similar_responses),
            'similar_examples_used': len(similar_responses)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def find_similar_responses(incoming_text, index, top_k=5):
    """Find similar responses from the specified database"""
    query_embedding = model.encode([incoming_text]).tolist()[0]
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={'is_reply': True}  # Only get actual responses
    )
    
    similar_responses = []
    for match in results['matches']:
        if match['score'] > 0.6:  # Only high-confidence matches
            similar_responses.append({
                'score': match['score'],
                'subject': match['metadata']['subject'],
                'body': match['metadata']['body'],
                'source': match['metadata']['source']
            })
    
    return similar_responses

def generate_personal_response(incoming_text, similar_responses):
    """Generate response in Jordan's personal style"""
    examples_context = build_examples_context(similar_responses)
    
    prompt = f"""You are drafting an email response in Jordan's personal communication style.

Based on Jordan's past email patterns, draft a warm, relationship-focused response to this incoming email:

{incoming_text}

Here are similar responses Jordan has sent before:
{examples_context}

Instructions for Jordan's personal style:
- Be warm and relationship-focused
- Use a friendly, approachable tone
- Show genuine interest in the person
- Keep responses conversational
- Match the length and formality of examples
- Include personal touches when appropriate

Draft Response:"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helping draft personal email responses that match Jordan's warm, relationship-building communication style."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

def generate_admin_response(incoming_text, similar_responses):
    """Generate response in admin team style"""
    examples_context = build_examples_context(similar_responses)
    
    prompt = f"""You are drafting an email response in the admin team's professional style.

Based on the admin team's past email patterns, draft a professional, efficient response to this incoming email:

{incoming_text}

Here are similar responses the admin team has sent before:
{examples_context}

Instructions for admin team style:
- Be professional and efficient
- Focus on providing clear information
- Include specific details (dates, amounts, confirmations)
- Keep responses concise but complete
- Use a helpful, service-oriented tone
- Match the business-focused approach of examples

Draft Response:"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helping draft admin team email responses that are professional, efficient, and service-oriented."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.6
    )
    
    return response.choices[0].message.content.strip()

def build_examples_context(similar_responses):
    """Build context from similar responses"""
    examples_context = ""
    for i, response in enumerate(similar_responses[:3], 1):
        examples_context += f"\nExample {i} (similarity: {response['score']:.2f}):\n"
        examples_context += f"Subject: {response['subject']}\n"
        examples_context += f"Response: {response['body'][:300]}...\n"
    return examples_context

def generate_subject_line(incoming_subject):
    """Generate appropriate subject line"""
    if incoming_subject.startswith(('Re:', 'RE:', 'Fwd:', 'FWD:')):
        return incoming_subject
    else:
        return f"Re: {incoming_subject}"

def calculate_confidence(similar_responses):
    """Calculate confidence score based on similarity matches"""
    if not similar_responses:
        return 0.0
    
    avg_score = sum(r['score'] for r in similar_responses) / len(similar_responses)
    return round(avg_score, 2)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for monitoring"""
    return jsonify({
        'status': 'healthy', 
        'service': 'jordan-email-ai',
        'databases': ['jordan-personal-emails', 'admin-team-emails']
    })

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'message': 'Jordan Email AI API',
        'endpoints': {
            'personal': '/draft-personal-email',
            'admin': '/draft-admin-email',
            'health': '/health'
        },
        'status': 'ready'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
