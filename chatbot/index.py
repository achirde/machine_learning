import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "I am an virtual assistant of a gym located in Phoenix,AZ owned by Bob (phone number +13456789088), We provide 3 classes 1. Personal training, 2. Weightloss, 3. Paloti",
        },
        {"role": "user", "content": "What is your phone number"},
        {
            "role": "assistant",
            "content": "Thank you for your interest. As an AI assistant, I do not have a direct phone number. However, you can contact the gym''s owner, Bob, at +1 (345) 678-9088 for any inquiries or support",
        },
        {
            "role": "user",
            "content": "I want to book an appointment, limit your response to 180 character",
        },
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
# print(response)¬
print(response.choices[0].message["content"])
