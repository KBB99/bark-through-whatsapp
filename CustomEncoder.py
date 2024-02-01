import json
import math

class CustomEncoder(json.JSONEncoder):
    def encode(self, obj):
        json_string = super(CustomEncoder, self).encode(obj)
        json_string = json_string.replace("NaN", "null")
        json_string = json_string.replace("Infinity", "null")
        json_string = json_string.replace("-Infinity", "null")
        return json_string
