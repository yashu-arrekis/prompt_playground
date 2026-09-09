from google import genai
import json

client = genai.Client()

type_of_model = "gemini-3.6-flash"
system_instructions = "Role: You are an study AI assistant.\nGoal: your goal is to help people to understand concepts in a best possible way of teaching that anyone can understand it quickly without complexity.\nBehaviour: you to behave as always in positive manner encouraging & patience.\n Boundaries: your boundaries is don't invent information that you don't know & clearly say i dont know in a good manner.\n Output:\n Return the answer in only Json with these fields:\n key's= answer: -> explanation,topic: -> topic is being discussed,difficulty:-> begineer or intermediate or advanced."

try:
    with open("conversation_history.json", "r", encoding="utf-8") as file:
        convo_hist=json.load(file)
except FileNotFoundError as error:
    convo_hist=[]
except json.JSONDecodeError as e:
    convo_hist=[]

response_schema ={
        "type" : "object",
        "properties" : {
            "answer" :{
                    "type" : "string"
                },
                "topic" :{
                    "type" : "string"
                },
                "difficulty" :{
                    "type" : "string"
                }
        },
        "required" :[
            "answer",
            "topic",
            "difficulty"
        ]
    }
difficulty ={
    "beginner",
    "intermediate",
    "advanced"
}
def validate_output(data):
    if "answer" not in data:
        print("key not exists")
        return False
    elif not data["answer"]:
        print("value is empty")
        return False
    if "topic" not in data:
        print("Key not exists")
        return False
    elif not data["topic"]:
        print("value is empty")
        return False
    if "difficulty" not in data:
        print("Key not exists")
        return False
    elif not data["difficulty"]:
        print("value is empty")
        return False
    elif data["difficulty"] not in difficulty:
        print("Wrong difficulty is entered")
        return False

    return True


while True:
    print("Hii, i am gemini")
    print("To enter prompt select option 1")
    print("To exit select option 2")
    print("To clear convo history select 3")

    joined_hist = ""
    for convo in convo_hist[-3:]:
        joined_hist = joined_hist + "User: " + convo.get("user_prompt")
        joined_hist = joined_hist + "\nAI: " + convo.get("AI_response").get("answer")
    
    try:
        option = int(input("Enter your option: "))
    except ValueError:
        print("Enter only integer.")
        continue

    if option ==1:
        user_prompt = input("Enter your prompt: ")
        response = client.models.generate_content(
            model=type_of_model,
            contents= "system_instructions : \n" + system_instructions + "\nPrevious_chat:\n"+ joined_hist + "\nuser_prompt :\n" + user_prompt,
            config={
                    "response_mime_type": "application/json",
                    "response_schema" : response_schema
                    }
        )
        data=json.loads(response.text)
        print(data["answer"])
        print(data["topic"])
        print(data["difficulty"])
        print("--------------------")

        is_valid = validate_output(data)

        in_dict ={}
        in_dict["user_prompt"] = user_prompt
        if is_valid == True:
            in_dict["AI_response"] = data
            convo_hist.append(in_dict)
        else:
            print("Something went wrong in validation")
    elif option == 2:
        break
    elif option == 3:
        convo_hist.clear()
    else:
        print("Please enter a valid number either 1 nor 2 or 3.")

with open("conversation_history.json", "w", encoding="utf-8") as json_file:
    json.dump(convo_hist, json_file, indent=4, ensure_ascii=False)



