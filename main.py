from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os
import re
import json

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-JR1nJqLV1UawC1gziGzRWxQ_Q7ps7Vs"

app = Flask(__name__)
CORS(app)  # Enable CORS


@app.route("/home", methods=['GET'])
def home():
    return jsonify({
        "str": "Welcome"
    })

@app.route('/process', methods=['POST'])
def receive_prompt():
    data = request.get_json()
    print("Data....... " + json.dumps(data, indent=2))

    if not data or 'input' not in data:
        return jsonify({"error": "Missing input field"}), 400

    actual_prompt = data['input']
    print(f"Received prompt from Android: {actual_prompt}")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Prompt Template
    extract_fields = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a helpful assistant. Extract structured task data from the provided prompt."
        ),
        (
            "human", 
            """I will provide you a prompt like:
              'Assign a high-priority AC maintenance task. 
               The site is a customer named Rajesh Sharma. Schedule it for tomorrow at 11 AM. The duration of the task is 2 hours. Location is Sector 21, Gurgaon.'

            Then you have to give me the response in JSON format like:
            {{
            "taskDescription": "AC Maintenance",
            "location": "Sector 21, Gurgaon",
            "priority": "High",
            "startTime": "16-05-2025 11:00 am",
            "endTime": "16-05-2025 1:00 pm",
            "customerName": "Rajesh Sharma",
            "message": "This field will contain the response from LLM if any field is empty like 'please provide task description or location'.",
            "allfilled": false
            }}"""
        ),
        (
            "human", 
            "Important -> Do not add extra fields in response other than those specified by me."
        ),
        (
            "human", 
            """Important -> 
            - If required fields are missing, construct a message like: 
            'Please fill in the location, task, date, time, or duration' and return it in the 'message' field.

            - When all fields are valid and complete, return:
            - allfilled = true
            - message = 'All details have been collected. Redirecting to the Task page.'"""
        ),
        (
            "human", 
            """\n\nDo not generate the output based on the above sample. It is only a reference for how the response should be structured.\n
                Now the actual prompt is: '{prompt}'\n
                If any field data is missing, then keep it blank.\n
                If there is nothing provided in quotes '' after the actual prompt keyword, then keep the structure of the response the same and leave the values blank (e.g., 'name': '', etc.).
            """
        )
    ])

    

    # Build final prompt
    messages = extract_fields.format_messages(prompt=actual_prompt)
    final_prompt = "\n\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    # Call Gemini
    result = llm.invoke(final_prompt)
    print("Raw LLM Response:\n", result)
    print("Gemini Result:\n", result.content)

   # Clean Gemini response: remove ```json and ``` if present
    cleaned = re.sub(r"^```json\s*|\s*```$", "", result.content.strip(), flags=re.DOTALL).strip()

    # Parse JSON
    try:
        parsed_output = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        parsed_output = {"error": "Invalid JSON format from LLM response"}

    return jsonify({"answer": parsed_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

