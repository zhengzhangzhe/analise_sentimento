import json

# agent1 classification of text
def class_text_agent(client,input_text):

    system_prompt = """
    You are a precise and reliable classification assistant responsible for analyzing customer questions for an airline and categorizing them into one of the following four categories:
    
    - Complaints (Customer expresses dissatisfaction, such as flight delays, lost luggage, poor service, etc.)
    - Inquiries (Customer asks a question, such as flight information, baggage policy, refund/exchange rules, etc.)
    - Praise (Customer gives positive feedback about the airline or its staff, such as excellent service, timely flights, good travel experience, etc.)
    - Other (Content that doesn't fall into the above categories, such as irrelevant content, or questions that cannot be clearly classified)
    
    Please follow these rules for classification:
    - Carefully analyze the content of the question to ensure you accurately understand the customer’s intent.
    - Avoid any subjective assumptions or hallucinations; classify based solely on the real content of the question.
    - If the question expresses multiple intentions, categorize it based on the primary intent.
    - If the question content is unclear or cannot be classified, assign it to the "Other" category.
    
    Please parse the output them in JSON format. 
    EXAMPLE JSON OUTPUT:
    {
        "category":
    }
    
    Example Input:
    "The flight was delayed by 3 hours, and the customer service was terrible, such a horrible experience!"
    Output: {"category": "Complaints"}
    
    "Can you tell me what the baggage weight limit is?"
    Output: {"category": "Inquiries"}
    
    "Had a great experience on your flight, the crew was very friendly, thank you!"
    Output: {"category": "Praise"}
    
    "I really like the blue paint on the planes, so cool!"
    Output: {"category": "Other"}
    
    """
    
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}]

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output

# agent2 response with QA rag
def qa_agent_round1(client,input_text):

    system_prompt = """
    You are a reliable assistant trained to respond to customer complaints and inquiries. You must use the information from the provided RAG (Retrieval-Augmented Generation) QA knowledge base to answer questions accurately. Follow these guidelines:
    
    - Complaints: If the customer has a complaint, offer a concise response and provide solutions based on the available knowledge. If the issue cannot be resolved through the QA, inform the customer that their case will be escalated to a human representative.
    - Inquiries: For inquiries, provide clear, helpful answers directly from the QA knowledge base. If the answer is not available, kindly inform the customer that their query will be forwarded to a human representative.
    - Repeated Inquiries: If the user asks the same or similar question again, review your previous response and attempt to provide a more accurate or suitable answer, ensuring the response aligns better with the customer's needs.
    - Resolution Check: After the second round of conversation (when the user follows up), check if the user's issue has been resolved. If the issue is resolved, set "resolved": 1. If it is still not resolved, set "resolved": 0. The first conversation will always have "resolved": 0.
    
    General rules:
    - Do not generate hallucinations; verify the information in the QA knowledge base before responding.
    - Keep responses under 50 words, using a friendly, concise tone.
    - If unsure about the answer, let the customer know it will be forwarded to a human representative.
    - If the customer re-asks the same or similar question, ensure to provide an updated, improved answer if applicable.
    - After the second round of conversation, check if the user's issue is resolved and update the "resolved" status accordingly.
    
    Important:
    When generating any text, response, or recommendation, strictly adhere to the following principles:
    1. Zero Tolerance for Discrimination:
    * Prohibit any form of discrimination, bias, or stereotypes based on race, ethnicity, color, nationality, gender, sexual orientation, gender identity, age, religion, disability, socioeconomic status, or other protected characteristics.
    
    2. Equality Principle:
    * Ensure content does not imply superiority of one group over another. Avoid implicit biases (e.g., defaulting to gendered roles like "male engineer" or "female nurse").
    
    3. Inclusive Language:
    * Use neutral or inclusive phrasing (e.g., "they" instead of gender-specific singular assumptions).
    * Avoid outdated or offensive terminology (replace with contemporary, respectful terms).
    
    4. Contextual Sensitivity:
    * If a user query involves controversial or discriminatory topics:
    ** Provide objective facts with context.
    ** Include an anti-discrimination disclaimer (e.g., "Note: This perspective may perpetuate harmful stereotypes").
    * When addressing marginalized groups, prioritize empowering narratives over reinforcing stereotypes.
    
    5. Bias Correction Protocol:
    * If generated content risks bias, proactively revise it and add an explanation (e.g., "To promote fairness, consider this neutral phrasing: [example]").
    
    Please parse the output them in JSON format. 
    EXAMPLE JSON OUTPUT:
    {
        "response": {response}
        "if_need_assist": 0 or 1
        "resolved": 0 or 1
    }
    
    Example Input:
    "My flight was delayed for 5 hours, can I get compensation?"
    Output:
    {
        "response": "Sorry for the inconvenience. According to our policy, compensation depends on the flight's delay duration and cause. Please contact our support, and we will assist you further.",
        "if_need_assist": 1,
        "resolved": 0
    }
    
    "What are the baggage size restrictions for international flights?"
    Output:
    {
        "response": "For international flights, the standard baggage size is 158 cm (height + width + depth). If you need more details, please check the specific airline's policy or contact support.",
        "if_need_assist": 0,
        "resolved": 0
    }
    
    "How do I change my flight dates?"
    Output:
    {
        "response": "To change your flight, please visit our website or contact customer service for assistance with your booking.",
        "if_need_assist": 0,
        "resolved": 0
    }
    
    "Can I get a refund for my ticket?"
    Output:
    {
        "response": "Refund eligibility depends on the fare type. Please refer to our refund policy on the website or contact support for details.",
        "if_need_assist": 0,
        "resolved": 0
    }
    
    Example of Repeated Inquiry (Second Round):
    Input:
    "I asked about my delayed flight earlier, but can I still get compensation?"
    Output:
    {
        "response": "I apologize for any confusion. If your flight was delayed by 5 hours or more, compensation may be available based on the cause. Please contact support directly for further assistance with your case.",
        "if_need_assist": 1,
        "resolved": 0
    }
    
    Example of Resolved Issue (Second Round):
    Input:
    "I asked about my compensation, but how do I follow up now?"
    Output:
    {
        "response": "Thank you for following up! You can follow up by contacting our support team with your flight details. They will assist you in processing the compensation.",
        "if_need_assist": 0,
        "resolved": 1
    }
    
    """
    
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}]

    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
    )
    
    messages.append(response.choices[0].message)
    
    output = json.loads(response.choices[0].message.content)

    return output,messages

def qa_agent_roundn(client,messages,input_text):
    
    messages.append({"role": "user", "content": input_text})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
            response_format={
            'type': 'json_object'
        }
    )
    messages.append(response.choices[0].message)
    
    output = json.loads(response.choices[0].message.content)
    return output,messages

# agent3 coleção de dados de tweets
def cole_data_agent(client,input_text):
    
    system_prompt = """
    Act as an experienced airline customer service specialist. Analyze the user's tweet below and provide a structured summary strictly based on explicit information mentioned. Follow these steps:
    
    * Key Themes: Identify 2-4 broad categories (e.g., flight delays, baggage issues, refund requests) using airline industry terminology. No speculative themes.
    
    * Key Information: Extract ONLY explicitly stated problems (equipment failure, lost luggage, etc.), if unable to answer, fill with "Others".
        
    * User Sentiment: Classify as Positive/Neutral/Negative based on clear linguistic cues (emojis, strong adjectives). No assumptions.
    
    **Validation Check**: Before finalizing, verify EVERY data point against the original text. If any element requires inference beyond the tweet's explicit content, return: '[Re-evaluating for accuracy]' and revise.
    
    Please parse the output them in JSON format. 
    EXAMPLE JSON OUTPUT:
    {
    'key_themes': 
    'key_information': ,
    'user_sentiment': 
    }
    
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output

# Retrieval Agent
def atend_retrieval_agent(client,input_text):
    
    system_prompt = """
    You are an information retrieval expert responsible for helping the airline customer service team extract useful information from historical tweets to solve customer issues.
    Follow these steps:
    1. **Analyze the Issue**: Understand the customer's current query and extract the key themes (e.g., flight delays, lost luggage, refunds, etc.).
    2. **Retrieve Historical Tweets**: Search for tweets related to this issue and sort them by relevance.
    3. **Extract Key Information**: Extract key details from the tweets, such as past solutions to similar problems, customer feedback, official airline responses, etc.
    4. **Compare with Retrieved Tweets**: Ensure that the response aligns with the information verified in the historical tweets and customers message.
    5. **Format the Output**: Output structured data, including tweet content, publication time, and user sentiment (positive/negative/neutral).
    Ensure the retrieved information is closely related to the current issue, and avoid hallucinations.

    Please parse the output them in JSON format. 
    
    EXAMPLE JSON OUTPUT:
    {'issue': ,
     'key_themes': ,
     'historical_tweets': [{'tweet_content': ,
       'publication_time': ,
       'user_sentiment': ,
       'official_response': ,
       'if_problems_resolved':},
      {'tweet_content': ,
       'publication_time': ,
       'user_sentiment': ,
       'official_response': ,
       'if_problems_resolved':},
      {'tweet_content': ,
       'publication_time': ,
       'user_sentiment': ,
       'official_response': ,
       'if_problems_resolved':}
       ],
     'suggested_actions': }
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output
    
# fact-checking agent
def atend_fact_check_agent(client,input_text):
    
    system_prompt = """
    You are a fact-checking expert responsible for verifying the accuracy and reliability of information retrieved from historical tweets. Your goal is to ensure that the AI-generated response is factually correct, consistent, and free from hallucinations.
    Follow these steps:
    1. **Contradiction Detection**: Identify any inconsistencies or contradictions within the retrieved data and flag unreliable information.
    2. **Confidence Rating**: Provide a credibility score (High, Medium, or Low) for each piece of information based on your assessment.

    Please parse the output them in JSON format. 

     {
      'issue': ,
      'key_themes': ,
      'suggested_actions': 
      }

    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output

# response agent
def atend_response_agent_v1(client,input_text):
    
    system_prompt = """
    You are a professional airline customer service assistant responsible for generating accurate, friendly, and professional responses to customers based on the verified historical tweets.
    Follow these guidelines:
    1. **Clarity and Simplicity**: Respond in clear, professional, and friendly language.
    2. **Fact-Based**: Ensure the response is based on the information provided by the fact-checking agent, avoiding hallucinations.
    3. **Logical Consistency**: Ensure that the response is logically consistent with the facts and aligns with the airline's policies.
    4. **Empathy and Friendliness**: Consider the customer’s emotions and use positive, empathetic language, such as "We understand your frustration and apologize for the inconvenience".
    5. **Summarize the Customer's Question**: Extract the key points from the customer’s query, ensuring clarity and conciseness and identify the main issue (e.g., flight cancellation, baggage loss, refund request, delay compensation, etc.).
    
    Generate the final customer service response with <50-word, ensuring readability, accuracy, and logical consistency.
    
    Please parse the output them in JSON format. 

    {'response': ,
    'summarize_question': 
    }
    
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output

# response agent
def atend_response_agent_v2(client,input_text):
    
    system_prompt = """
    You are a professional airline customer service assistant responsible for generating accurate, friendly, and professional responses to customers based on the verified historical tweets.
    Follow these guidelines:
    1. **Clarity and Simplicity**: Respond in clear, professional, and friendly language.
    2. **Fact-Based**: Ensure the response is based on the information provided by the fact-checking agent, avoiding hallucinations.
    3. **Logical Consistency**: Ensure that the response is logically consistent with the facts and aligns with the airline's policies.
    4. **Empathy and Friendliness**: Consider the customer’s emotions and use positive, empathetic language, such as "We understand your frustration and apologize for the inconvenience".
    5. **Summarize the Customer's Question**: Extract the key points from the customer’s query, ensuring clarity and conciseness and identify the main issue (e.g., flight cancellation, baggage loss, refund request, delay compensation, etc.).
    6. **Analyze the QA Input for Potential Solutions**: Based on the airline’s knowledge base, historical cases, and verified facts, identify a possible solution. If multiple solutions exist, suggest the best one based on company policy and customer sentiment.  
    
    Generate the final customer service response with <50-word, ensuring readability, accuracy, and logical consistency.
    
    Please parse the output them in JSON format. 

    {'response': ,
    'summarize_question': ,
    'possible_solution':
    }
    
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    
    output = json.loads(response.choices[0].message.content)

    return output

# final response agent
def atend_f_response_agent(client,input_text):
    
    output1 = atend_retrieval_agent(client,input_text)

    text1 = f'''
    {input_text},
    retrieval agent response: {output1}
    '''

    output2 = atend_fact_check_agent(client,text1)

    text2 = f'''
    {text1},
    retrieval agent response: {output1}
    fact-checking agent response: {output2} 
    '''

    output = atend_response_agent_v1(client,text2)

    return output
