# app.py - Ultra-simple Flask API
from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# Initialize OpenAI - moved inside routes to avoid startup issues
def get_openai_client():
    return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/draft-personal-email', methods=['POST'])
def draft_personal_email():
    """Draft email using Jordan's personal style"""
    try:
        data = request.json
        incoming_subject = data.get('subject', '')
        incoming_body = data.get('body', '')
        sender_email = data.get('from', '')
        
        prompt = f"""You are Jordan Nielsen, drafting a personal email response in your warm, relationship-focused style.

Incoming email:
From: {sender_email}
Subject: {incoming_subject}
Body: {incoming_body}

Draft a response that matches Jordan's communication style:
- Warm and friendly tone
- Relationship-focused
- Conversational and genuine
- Shows personal interest
- Professional but approachable

Response:"""

        response = get_openai_client().chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Jordan Nielsen, responding to emails in your personal, warm communication style based on years of relationship-building emails."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        draft_response = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'style': 'personal',
            'draft_subject': f"Re: {incoming_subject}" if not incoming_subject.startswith('Re:') else incoming_subject,
            'draft_body': draft_response,
            'confidence_score': 0.9
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/draft-admin-email', methods=['POST'])
def draft_admin_email():
    """Draft email using admin team style"""
    try:
        data = request.json
        incoming_subject = data.get('subject', '')
        incoming_body = data.get('body', '')
        sender_email = data.get('from', '')
        
        prompt = f"""You are responding as part of the admin team, drafting a professional email response.

Incoming email:
From: {sender_email}
Subject: {incoming_subject}
Body: {incoming_body}

Draft a response that matches the admin team's communication style:
- Professional and efficient
- Clear and concise
- Service-oriented and helpful
- Include specific details (dates, amounts, confirmations when relevant)
- Business-focused but friendly

Response:"""

        response = get_openai_client().chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are part of an admin team, responding to emails in a professional, efficient, and service-oriented style."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.6
        )
        
        draft_response = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'style': 'admin',
            'draft_subject': f"Re: {incoming_subject}" if not incoming_subject.startswith('Re:') else incoming_subject,
            'draft_body': draft_response,
            'confidence_score': 0.9
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'jordan-email-ai-basic',
        'version': '3.0'
    })

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'message': 'Jordan Email AI API - Basic Version',
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
