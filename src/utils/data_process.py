import ast
import csv
import json
import os
import re
from src.interview_session.session_models import Message



def save_feedback_to_csv(interviewer_message: Message, feedback_message: Message, user_id: str, session_id: str):
    """Save feedback message to a CSV file with the last conversation message"""

    # Prepare the feedback directory
    feedback_dir = os.path.join(os.getenv("LOGS_DIR", "logs"), user_id, 'feedback')
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, f'session_{session_id}.csv')

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
            writer.writerow(['timestamp', 'interviewer_message', 'user_feedback'])

    # Clean and prepare the messages
    interviewer_content = interviewer_message.content if interviewer_message else ''
    feedback_content = feedback_message.content if feedback_message else ''
    
    # Append the feedback
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
        writer.writerow([
            feedback_message.timestamp.isoformat(),
            interviewer_content,
            feedback_content
        ])
        
def read_from_pdf(file_path: str):
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def safe_parse_json(text: str):
    text = text.strip()
    if not text:
        return None

    # Try to find ```json ... ``` fenced block first
    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fallback later

    # Try parsing entire text as JSON directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (last resort)
    try:
        parsed_dict = ast.literal_eval(text)
        if isinstance(parsed_dict, dict):
            return parsed_dict
    except Exception:
        pass
    
    return None
    