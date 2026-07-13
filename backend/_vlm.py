import os 
import groq
from groq import Groq
import time
import base64
import dotenv 
from dotenv import load_dotenv
load_dotenv()

# unknown person detection (local) -> get CLIP to describe person

animal_question_prompt = """What animal are you seeing? Return only the animal name, but make it specific. 
Specific as in what breed or species it is, but not so specific (something a taxonomist would classify it as). 

For example: 
- Dog -> Golden Retriever 
- Bear -> Brown Bear 

You are also able to mention size. 

Return only the animal name (and other details if they are relevant) in less than 20 words. No additional text, no thinking tags, no json strings - nothing else. 
"""
person_description_prompt = """You are a home security agent. Describe what you see outside. 
Describe: 
- What the person looks like 
- All items in the scene (what they are carrying)
- Any other relevant details in a home security setting 

Start your answer with "There is a ____ at the door" and then continue with your description of them. 

Your description must be in a single, short sentence. Only include the details relevant to a home security setting (make your explanation general, 
and specific when needed (if the specific details are very important)).

Your sentence must be no longer than 25 words. 

For example, if a man was at the door holding a weapon, your response should be: 

There is a [GENERAL DESCRIPTION OF PERSON: SKIN COLOR, BUILD, WHAT THEY ARE WEARING] at the door with [OBJECT_DESCRIPTION]. 

INCLUDE NOTHING ELSE IN YOUR RESPONSE. JUST THE DESCRIPTION OF THE PERSON, AND THE ITEMS THEY ARE CARRYING. 

DO NOT INFER INTENT OF ACTIONS OR EMOTIONS. USE GENERAL TERMS FOR OBJECTS. 
"""

bird_image = "/Users/sv/Downloads/birdie.webp"
person_image = "download_audio.jpg"

import time 
def infer_gemini():
    from google import genai
    from google.genai import types

    # Initialize the client
    client = genai.Client(api_key = os.getenv("google_api_key"))

    # Read your image file
    with open(person_image, "rb") as f:
        image_bytes = f.read()

    # Send the request
    start_time = time.time()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            person_description_prompt
        ]
    )

    print(response.text)
    print(f"Time taken: {time.time() - start_time}")

#infer_gemini()

def encode_image(image_path: str):
    with open(image_path, 'rb') as image_file: 
        return base64.b64encode(image_file.read()).decode('utf-8')
    

## gemini - 2 - 6 - 12 seconds
# groq - < 1 second

def infer_groq():
    client = Groq(api_key = os.getenv("groq_api_key"))
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": person_description_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(person_image)}"
                }
            }
            ]
        }
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

start_time = time.time()
infer_groq()
print(f"Time taken: {time.time() - start_time}")

# import base64
# from openai import OpenAI

# client = OpenAI(api_key = os.getenv("openai_api_key"))

# def encode_image(path: str) -> str:
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")


# details = ['low', 'high', 'original']
# for detail in details:
#     start_time = time.time()
#     resp = client.responses.create(
#     model="gpt-5.4-nano",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": person_description_prompt},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{encode_image(person_image)}",
#                     "detail": detail,
#                 },
#             ],
#         }
#     ],
# )

#     print(f'Detail {detail}: \n{resp.output_text}')
#     print(f"Time taken for {detail}: {time.time() - start_time}")
#     print("\n")
