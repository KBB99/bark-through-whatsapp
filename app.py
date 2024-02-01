from flask import Flask, request
import process_message
from flask_cors import CORS
from CustomEncoder import CustomEncoder
import json
import openai
import requests
import tempfile

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello world!</p>"

import tempfile

@app.route("/process_whats_app_message", methods=['POST'])
def process_whats_app_message():

    media_url = request.form.get("MediaUrl0")
    if media_url:
        print("Received media url:", media_url)
        # Download the media file to local storage
        media_file_content = requests.get(media_url).content
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg", mode='wb+') as tmp_media_file:
            tmp_media_file.write(media_file_content)
            tmp_media_file.flush()  # Make sure all content is written
            tmp_media_file.seek(0)  # Reset file position to beginning
            # Transcribe the audio
            message_content = openai.Audio.transcribe("whisper-1", tmp_media_file).text
    else:
        # Get the message content from text
        message_content = request.form.get("Body")

    # Print the message content
    print("Received message:", message_content)
    user_number = request.form.get("From")
    print("Received from:", user_number)
    
    process_message.process(message_content, user_number)
    return "Success"


@app.route("/process_user_message")
def process_user_message():
    query = request.args.get('query','')
    user_phone_number = request.args.get('user_phone_number','')
    print(query, user_phone_number)
    response = process_message.process(query, user_phone_number)
    return json.dumps(response, cls=CustomEncoder)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)