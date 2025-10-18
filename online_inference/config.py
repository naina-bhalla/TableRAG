import os
v3_config = {
    "url": "https://api.cohere.ai/v1/generate",
    "model": "command",
    "api_key": os.getenv("COHERE_API_KEY"),
}

sql_service_url = 'http://127.0.0.1:5000/' 


config_mapping = {
    "v3": v3_config
}