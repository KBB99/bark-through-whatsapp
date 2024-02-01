from twilio.rest import Client

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

message = client.messages.create(
    # body='Here is your response.',
    media_url=['https://omega-testing-2022.s3.amazonaws.com/output/response-19185a7c-a1b4-475a-a715-07689e3bf257.wav'],
    from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
    to='whatsapp:+1234567890'      # The recipient's number
)

print(message.sid)
