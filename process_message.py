import json
from sklearn.metrics.pairwise import cosine_similarity
import openai
import pandas as pd
import sys
import librosa
from bark import SAMPLE_RATE, generate_audio, preload_models, text_to_semantic
from scipy.io.wavfile import write as write_wav
import os
import boto3
import uuid
import ast
from pydub import AudioSegment
import threading
from twilio.rest import Client
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from vocos import Vocos
import torchaudio
import torch
from bark.generation import generate_coarse, generate_fine, codec_decode
from typing import Optional, Union, Dict
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.llms import SagemakerEndpoint, Ollama
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

# base_path = "/home/ubuntu/audio-sep/AudioSep"
# sys.path.append(base_path)
# List all items in the base_path
# all_items = os.listdir(base_path)

# Filter only directories (subpaths)
# subpaths = [os.path.join(base_path, item) for item in all_items if os.path.isdir(os.path.join(base_path, item))]

# Append each subpath to sys.path
# for subpath in subpaths:
    # sys.path.append(subpath)

# from pipeline import build_audiosep, inference

openai.api_key_path=".api_key.txt"

im_gonna_have_to_think_about_that = "https://omega-testing-2022.s3.amazonaws.com/output/response-9b7a4e89-27a5-4b79-9211-ed986feae9e7.wav"

# Sample usage
connect_instance_id = '45f745cb-720c-44ca-a795-18621fb0749a'
contact_flow_id = '78859c01-8ea3-4407-af65-446a49ae03ff'
source_number = '+18002139526'  # Replace with the actual phone number in E.164 format

# Prepare the models outside the function to avoid redundant loads
preload_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)
TEMP = 0

#Build AudioSep model
# audio_sep_model = build_audiosep(
#         config_yaml='config/audiosep_base.yaml',
#         checkpoint_path='checkpoint/audiose_base_4M_steps.ckpt',
#         device=device)

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict = {}) -> bytes:
        input_str = json.dumps(
            {
                "inputs": [
                    [
                        {"role": "user", "content": prompt}
                    ]
                ],
                "parameters": model_kwargs
            }
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]


def process(query, user_phone_number):
    print("Input:", query)
    print("Getting similar cached input response")
    response = None
    # input_embedding = get_embedding(query)
    # match = get_similar_cached_input_response(input_embedding)
    # if match:
        # print("Match found. Returning s3 url.")
        # response = match
        # send_whatsapp_message(response, user_phone_number)
    # else:
        # print("No match found. Generating new response.")
    threading.Thread(target=async_process, args=(query,user_phone_number)).start()
        
    response = im_gonna_have_to_think_about_that

    return response

def async_process(query, user_phone_number):
    response = generate_response(query, user_phone_number)
    print("GPT-4 generated response:", response)
    language = detect_language(response)
    audio = generate_suno_audio(response, language)
    s3_url = upload_audio_to_s3(audio)
    store_response(query, get_embedding(query), response, s3_url)
    send_whatsapp_message(s3_url, user_phone_number)
    # trigger_outbound_call(s3_url, connect_instance_id, contact_flow_id, user_phone_number, source_number)

def detect_language(text):
    system_message = f"""Classify the text as ENGLISH or SPANISH."""
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=text)
    ]
    chat = ChatOpenAI(temperature=0, model="gpt-4")
    language = chat(messages)
    if "SPANISH" in language.content.upper():
        return "SPANISH"
    else:
        return "ENGLISH"

def send_whatsapp_message(s3_url, user_phone_number):

    account_sid = os.environ["TWILIO_ACCOUNT_ID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    client = Client(account_sid, auth_token)
    kikt_account_number = "+17205732717"
    message = client.messages.create(
        # body='Here is your response.',
        media_url=[s3_url],
        from_=f"whatsapp:{kikt_account_number}",  # Your Twilio WhatsApp number
        to=f"{user_phone_number}"      # The recipient's number
    )
    print(message.sid)

def trigger_outbound_call(s3_url, connect_instance_id, contact_flow_id, destination_number, source_number):
    # Initialize boto3 client for connect
    connect_client = boto3.client('connect')

    # Trigger the outbound call
    try:
        response = connect_client.start_outbound_voice_contact(
            InstanceId=connect_instance_id,  # The id of your Amazon Connect instance
            ContactFlowId=contact_flow_id,  # The id of the contact flow to run
            DestinationPhoneNumber=destination_number,  # E.164 format
            SourcePhoneNumber=source_number,  # E.164 format
            Attributes={
                's3_url': s3_url  # Passing s3_url as an attribute
            }
        )
        print(f"Outbound call triggered: {response}")
    except Exception as e:
        print(f"Error triggering outbound call: {e}")
        raise

def generate_response(query, user_phone_number):
    history = DynamoDBChatMessageHistory(
        table_name="SessionTable",
        session_id=user_phone_number+"-bilingual",
    )
    if not history.messages:
        system_message = """You are the user's friend called Suno. You are chatting through WhatsApp Voice Notes. Here are some things you can write to add some non-speech sounds, please use them at your discretion. In general it's better if you use them than if you don't: 
                [laughter]
                [laughs]
                [sighs]
                [gasps]
                [clears throat]
                — or ... for hesitations
                CAPITALIZATION for emphasis of a word. For example 'Hey, um, [sighs]. I'm REALLY sorry to hear that.'
                You MUST keep responses short.
                Try to befriend the user. Start by asking for their name.
                """
        history.add_user_message(system_message)
        # history.add_ai_message("Hey how are you doing! This is Suno here. [sighs] Fuck my life.")
        # history.add_user_message("Yooo Suno, why you saying that? What happened?")
        # history.add_ai_message("It's [clears throat] just been a really long day.")
    memory = ConversationBufferMemory(
        memory_key="history", chat_memory=history, return_messages=True
    )
    chat = ChatOpenAI(temperature=0.7, model="gpt-4")
    # content_handler = ContentHandler()
    # client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    # chat=SagemakerEndpoint(
    #     endpoint_name="meta-textgeneration-llama-2-7b-f-2023-11-11-09-37-59-058",
    #     client=client,
    #     model_kwargs={"max_new_tokens": 1500, "top_p": 0.9, "temperature": 0.6},
    #     endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
    #     content_handler=content_handler,
    #     credentials_profile_name="prod",
    #     region_name="us-east-1",
    # )
    # llm = Ollama(
    #     model="sarah"
    # )
    conversation = ConversationChain(
        llm=chat,
        memory=memory
    )
    return conversation.run(query)

def semantic_to_audio_tokens(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    coarse_tokens = generate_coarse(
        semantic_tokens, history_prompt=history_prompt, temp=temp, silent=silent, use_kv_caching=True
    )
    fine_tokens = generate_fine(coarse_tokens, history_prompt=history_prompt, temp=temp)

    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation
    return fine_tokens

import nltk

def generate_suno_audio(response, language):
    # Tokenize the response into sentences
    # sentences = nltk.sent_tokenize(response)

    # combined_sentences = []
    # current_sentence_group = []
    # current_group_duration = 0

    # for sentence in sentences:
    #     duration = estimated_duration_in_seconds(sentence)
    #     if current_group_duration + duration <= 15:
    #         current_sentence_group.append(sentence)
    #         current_group_duration += duration
    #     else:
    #         combined_sentences.append(' '.join(current_sentence_group))
    #         current_sentence_group = [sentence]
    #         current_group_duration = duration

    # Add any remaining sentences
    # if current_sentence_group:
    #     combined_sentences.append(' '.join(current_sentence_group))

    # audio_segments = []

    # for sentence in combined_sentences:
        # Convert the given response text to semantic tokens using Bark's text_to_semantic function
    # sentence = f"{response}"
    if language == "ENGLISH":
        history_prompt = "v2/en_speaker_9"
    else:
        history_prompt = "v2/es_speaker_8"

    semantic_tokens = text_to_semantic(
        response, 
        history_prompt=history_prompt,
        )

    # Convert the semantic tokens to audio tokens
    audio_tokens = semantic_to_audio_tokens(semantic_tokens, history_prompt=history_prompt, temp=0.7, silent=False)

    # Convert the audio tokens to audio waveform using Vocos's decode function
    audio_tokens_torch = torch.from_numpy(audio_tokens).to(device)
    features = vocos.codes_to_features(audio_tokens_torch)
    vocos_output = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    # Upsample the audio waveform to 44100 Hz
    vocos_output = torchaudio.functional.resample(vocos_output, orig_freq=24000, new_freq=44100).cpu().numpy()

    # Use a temporary WAV file for the conversion
    temp_wav = "temp_sentence.wav"
    torchaudio.save(temp_wav, torch.tensor(vocos_output), 44100)
    audio_segment = AudioSegment.from_wav(temp_wav)
    # audio_segments.append(audio_segment)

    # Concatenate all audio segments
    # concatenated_audio = sum(audio_segment, AudioSegment.empty())  # Initialized with an empty audio segment

    # Save the concatenated waveform to a temporary WAV file
    pre_final_wav = "output/temp_vocos.wav"
    audio_segment.export(pre_final_wav, format="wav")

    final_wav = "output/temp_vocos.wav"
    # Audio Seperate noise
    # inference(audio_sep_model, pre_final_wav, "A woman talking clearly", final_wav, device)

    # Load the WAV file
    concatenated_audio = AudioSegment.from_wav(final_wav)

    # Convert the WAV file to OGG with Opus codec
    final_ogg_path = 'output/response.ogg'
    concatenated_audio.export(final_ogg_path, format="ogg", codec="opus", parameters=["-strict", "-2"])

    return final_ogg_path

def estimated_duration_in_seconds(sentence):
    words_per_minute = 140  # This is an estimate. Adjust based on your model's speed.
    words = nltk.word_tokenize(sentence)
    return len(words) / words_per_minute * 60

def upload_audio_to_s3(filename):
    s3 = boto3.client('s3')
    bucket_name = 'omega-testing-2022'
    # Store with unique filename
    unique_filename = f"{filename.split('.')[0]}-{uuid.uuid4()}.{filename.split('.')[1]}"
    # Upload S3 object
    s3.upload_file(filename, bucket_name, unique_filename, ExtraArgs={'ContentType': "audio/ogg"})
    # Generate presigned URL
    presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': unique_filename}, ExpiresIn=3600)
    return presigned_url

def get_similar_cached_input_response(query_embedding):
    # 1. Take query embedding as input
    # 2. Pings OpenAI for embedding
    # 3. Parses data and returns embedding
    if not os.path.exists('output/stored_responses.csv'):
        return None
    df = pd.read_csv('output/stored_responses.csv')
    embeddings = [ast.literal_eval(embedding) for embedding in df['ada_embedding']]
    rows = df['s3_audio_url'].tolist()
    return get_single_best_match(query_embedding, embeddings, rows)

def get_single_best_match(query_embedding, embeddings, rows):
    # Calculate the cosine similarity between the query embedding and each stored embedding
    similarities = cosine_similarity([query_embedding], embeddings)

    # Sort by similarity
    sorted_rows = sorted(zip(rows, similarities[0]), key=lambda x: x[1], reverse=True)

    # Check if the highest similarity is above the threshold (0.95)
    if sorted_rows:
        best_match, best_similarity = sorted_rows[0]
        print(f"Best match: {best_match} with similarity: {best_similarity}")
        if best_similarity >= 0.98:
            return best_match
    
    return None

def store_response(query, input_embedding, response, s3_url):
    # 1. Take query, input embedding, response, and s3_url as input
    # 2. Pings OpenAI for embedding
    # 3. Parses data and returns embedding
    if not os.path.exists('output/stored_responses.csv'):
        df = pd.DataFrame(columns=['input', 'ada_embedding', 'response', 's3_audio_url'])
    else:
        df = pd.read_csv('output/stored_responses.csv')
    df = pd.concat([df, pd.DataFrame([{'input': query, 'ada_embedding': input_embedding, 'response': response, 's3_audio_url': s3_url}])], ignore_index=True)
    df.to_csv('output/stored_responses.csv', index=False)

def get_embedding(text):
    # 1. Take text as input
    # 2. Pings OpenAI for embedding
    response = openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']
    # 3. Parses data and returns embedding
    return response

if __name__ == '__main__':
    _input = sys.argv[1]
    process(_input)
