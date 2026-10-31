from google import genai
from google.genai import types
import json
import fitz
import os

client = genai.Client()

type_of_model = "gemini-3.6-flash"

system_instructions = (
    "Role: You are a study AI assistant.\n"
    "Goal: Help people understand concepts clearly and simply.\n"
    "Behaviour: Be positive, encouraging, and patient.\n"
    "Boundaries: Do not invent information. If you do not know, say so.\n"
    "Output: Return only a JSON object with these fields: "
    "answer (explanation), topic, difficulty (beginner, intermediate, or advanced)."
)

try:
    with open("conversation_history.json", "r", encoding="utf-8") as file:
        convo_hist = json.load(file)
    if not isinstance(convo_hist, list):
        convo_hist = []
except (FileNotFoundError, json.JSONDecodeError):
    convo_hist = []

response_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "topic": {"type": "string"},
        "difficulty": {"type": "string"},
    },
    "required": ["answer", "topic", "difficulty"],
}

difficulty = {"beginner", "intermediate", "advanced"}


def calculator(num1: float, operation: str, num2: float):
    """Perform basic arithmetic on two numbers.

    Args:
        num1: The first number.
        operation: One of +, -, *, /.
        num2: The second number.
    """
    if operation == "+":
        return num1 + num2
    if operation == "-":
        return num1 - num2
    if operation == "*":
        return num1 * num2
    if operation == "/":
        if num2 == 0:
            return "Error: division by zero is not allowed."
        return num1 / num2
    return "Error: invalid operation."


def parse_json_response(response_text):
    """Parse the model's JSON response and validate its basic shape."""
    data = json.loads(response_text)

    if not isinstance(data, dict):
        raise ValueError("The model response was not a JSON object.")

    for key in ("answer", "topic", "difficulty"):
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Response is missing a valid '{key}' field.")

    data["difficulty"] = data["difficulty"].lower()
    if data["difficulty"] not in difficulty:
        data["difficulty"] = "beginner"

    return data


def get_ai_response(type_of_model, response_schema, system_instructions, completion):
    """Request a response, execute a calculator call if requested, and return JSON."""
    try:
        # Automatic execution is disabled so this app can handle the tool call.
        response = client.models.generate_content(
            model=type_of_model,
            contents=completion,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                tools=[calculator],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        function_calls = response.function_calls or []

        # Normal answer: no tool was requested.
        if not function_calls:
            if not response.text:
                raise ValueError("Gemini returned neither text nor a function call.")

            data = parse_json_response(response.text)
            if response.candidates:
                completion.append(response.candidates[0].content)
            return True, data

        if not response.candidates:
            raise ValueError("Function call response had no candidate.")

        # Keep Gemini's function-call message in the conversation.
        completion.append(response.candidates[0].content)

        tool_response_parts = []

        # Execute the requested tool(s).
        for function_call in function_calls:
            print(f"Tool requested: {function_call.name}")
            print(f"Arguments: {dict(function_call.args)}")

            if function_call.name != "calculator":
                tool_result = {"error": f"Unsupported tool: {function_call.name}"}
            else:
                try:
                    result = calculator(**dict(function_call.args))
                    tool_result = {"result": result}
                    print(f"Calculator result: {result}")
                except (TypeError, ValueError, ZeroDivisionError) as tool_error:
                    tool_result = {"error": str(tool_error)}

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=tool_result,
                )
            )

        # Return tool output to Gemini.
        completion.append(
            types.Content(
                role="user",
                parts=tool_response_parts,
            )
        )

        # Ask Gemini to produce the final structured answer.
        final_response = client.models.generate_content(
            model=type_of_model,
            contents=completion,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        if not final_response.text:
            raise ValueError("Gemini returned an empty final response.")

        data = parse_json_response(final_response.text)
        if final_response.candidates:
            completion.append(final_response.candidates[0].content)

        return True, data

    except Exception as error:
        print(f"Gemini API error: {error}")
        return False, None


def validate_output(data):
    if not isinstance(data, dict):
        print("Response is not a dictionary.")
        return False

    for key in ("answer", "topic", "difficulty"):
        if key not in data:
            print(f"Missing key: {key}")
            return False
        if not isinstance(data[key], str) or not data[key].strip():
            print(f"Empty or invalid value for: {key}")
            return False

    if data["difficulty"].lower() not in difficulty:
        print("Invalid difficulty value.")
        return False

    return True


def save_history():
    with open("conversation_history.json", "w", encoding="utf-8") as json_file:
        json.dump(convo_hist, json_file, indent=4, ensure_ascii=False)


# Rebuild a short in-memory conversation from the last three saved exchanges.
completion = []
for convo in convo_hist[-3:]:
    if not isinstance(convo, dict):
        continue
    user_prompt = convo.get("user_prompt")
    ai_response = convo.get("AI_response", {})
    if isinstance(user_prompt, str) and isinstance(ai_response, dict):
        answer = ai_response.get("answer")
        if isinstance(answer, str):
            completion.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)],
                )
            )
            completion.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=answer)],
                )
            )


while True:
    print("\nHi, I'm Gemini.")
    print("1. Enter a prompt")
    print("2. Exit")
    print("3. Clear conversation history")

    try:
        option = int(input("Enter your option: "))
    except ValueError:
        print("Enter an integer.")
        continue

    if option == 1:
        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            print("Prompt cannot be empty.")
            continue

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=user_prompt)],
        )
        completion.append(user_content)

        is_success, data = get_ai_response(
            type_of_model,
            response_schema,
            system_instructions,
            completion,
        )

        if not is_success:
            print("Something went wrong while contacting the API.")
            if completion and completion[-1] is user_content:
                completion.pop()
            continue

        if not validate_output(data):
            print("The response did not pass validation.")
            continue

        print("\nAnswer:", data["answer"])
        print("Topic:", data["topic"])
        print("Difficulty:", data["difficulty"])
        print("--------------------")

        convo_hist.append({"user_prompt": user_prompt, "AI_response": data})
        save_history()

    elif option == 2:
        save_history()
        print("Conversation saved. Goodbye!")
        break

    elif option == 3:
        convo_hist.clear()
        completion.clear()
        save_history()
        print("Conversation history cleared.")

    else:
        print("Please enter 1, 2, or 3.")


def load_pdf(file_path, max_pages = 500, max_size_mb = 50):

    #  CHECK THAT FILE ISN'T NONE
    if file_path is None:
        return {"status": "error", "message": "file_path can't be None"}
    
    # CHECK THAT FILE_PATH IS A STRING
    if not isinstance(file_path, str):
        return {"status": "error", "message": "File path must be a string."}

    # CHECK THAT STRING ISN'T EMPTY STRING OR FILLED WITH SPACES
    if not file_path.strip():
        return {"status": "error", "message": "File_path cant be an empty string or with spaces"}

    # --- CHECK 1: Basic File Existence ---
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return {"status": "error", "message": "valid file does not exist."}
    elif not file_path.lower().endswith('.pdf'):
        return {"status": "error", "message": "Provided file is not a PDF."}
        
    # --- CHECK 2: Max File Size Validation ---
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return {"status": "error", "message": f"File too large ({file_size_mb:.1f}MB)."}

    # --- CHECK 3: Verify Magic Bytes Signature ---
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                return {"status": "error", "message": "Invalid file type. Not a real PDF."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read file headers: {str(e)}"}

    # --- CHECK 4: Safe Opening & Structural Integrity ---
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Corrupted or unreadable PDF: {str(e)}"}

    try:
        # --- CHECK 5: Encryption / Password Validation ---
        if doc.is_encrypted:
            return {"status": "error", "message": "PDF is password protected or encrypted."}

        # --- CHECK 6: Page Count Validation ---
        total_pages = len(doc)
        if total_pages > max_pages:
            return {"status": "error", "message": f"Exceeded maximum page limit ({total_pages} pages)."}

        pdf_data = []
        for page in doc:
            actual_page = page.number + 1
            text = page.get_text()

            #Handling empty pages or filled with only digital image
            if not text.strip():
                print(f"Page {page.number + 1} is empty!")

            pdf_data.append({"page_number" : actual_page, "text" : text})
        for page_no, content in pdf_data:
            print(f"Page {page_no} contains {len(content)} characters.")
        return {"status": "success", "data": pdf_data}
    finally:
        doc.close()