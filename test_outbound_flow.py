import boto3

s3_url = "https://omega-testing-2022.s3.amazonaws.com/output/response-316ca342-23e0-4e13-8ad7-385ea110d4cb.wav"
# Sample usage
connect_instance_id = '45f745cb-720c-44ca-a795-18621fb0749a'
contact_flow_id = '78859c01-8ea3-4407-af65-446a49ae03ff'
source_number = '+18002139526'  # Replace with the actual phone number in E.164 format

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

trigger_outbound_call(s3_url, connect_instance_id, contact_flow_id, '+1234567890', source_number)