import os
import json
from datasets import load_dataset
    

"""
Following examples are produced by GPT-4o in order to make model
more aware about multiple-turn dialogues.
"""
gpt4o_generated_extra_turns = [
    # 10 Examples with 3 turns
    ["What are the main differences between HTTP and HTTPS?", 
     "HTTP stands for HyperText Transfer Protocol, while HTTPS is the secure version of it, using SSL/TLS encryption to protect data during transmission.", 
     "So, should all websites use HTTPS?", 
     "Yes, especially those handling sensitive data like login credentials or payment information, to ensure security and privacy."],

    ["How does a blockchain work?", 
     "A blockchain is a decentralized digital ledger that records transactions across multiple computers, ensuring security and transparency.", 
     "Why is decentralization important?", 
     "Decentralization prevents single points of failure, making the system more resilient to attacks and censorship."],

    ["Why do you think people procrastinate?", 
     "Procrastination can occur due to fear of failure, lack of motivation, or feeling overwhelmed by the task at hand.", 
     "How can we avoid procrastination?", 
     "Breaking tasks into smaller, manageable steps and setting deadlines can help overcome procrastination."] ,

    ["How do seasons affect agriculture?", 
     "Seasons affect agriculture by determining planting and harvesting times, influencing crop yields, and determining the types of crops that can be grown.", 
     "What crops are best for colder climates?", 
     "In colder climates, crops like wheat, barley, and certain root vegetables thrive, as they can withstand cooler temperatures."] ,

    ["What is the impact of social media on mental health?", 
     "Excessive use of social media can lead to feelings of isolation, anxiety, and depression, but it can also provide a sense of connection and support.", 
     "What can be done to use social media healthily?", 
     "Setting boundaries, limiting screen time, and engaging in offline activities are essential to maintaining mental health in the digital age."],

    ["Can you explain machine learning?", 
     "Machine learning is a subset of artificial intelligence that allows computers to learn from data without explicit programming.", 
     "What is the difference between supervised and unsupervised learning?", 
     "In supervised learning, the model is trained with labeled data, while unsupervised learning works with unlabeled data to find patterns and relationships."],

    ["Why is the speed of light constant?", 
     "The speed of light is constant in a vacuum because of the properties of electromagnetic waves and the nature of space-time itself.", 
     "Can anything travel faster than light?", 
     "According to current theories, nothing can travel faster than light, as it would require infinite energy to do so."] ,

    ["What role do probiotics play in gut health?", 
     "Probiotics are beneficial bacteria that aid in digestion, help balance gut microbiota, and strengthen the immune system.", 
     "How can probiotics be incorporated into a diet?", 
     "Yogurt, kefir, sauerkraut, and kimchi are excellent sources of probiotics, which can be added to the daily diet."] ,

    ["What factors influence stock market prices?", 
     "Stock prices are influenced by company performance, economic factors, market sentiment, and external events such as political or global crises.", 
     "Is the stock market inherently risky?", 
     "Yes, but with a diversified portfolio and a long-term investment strategy, the risk can be reduced significantly."] ,

    ["How can AI improve healthcare?", 
     "AI can analyze medical data for diagnosis, predict disease outbreaks, personalize treatment plans, and assist with drug discovery.", 
     "Are there ethical concerns with AI in healthcare?", 
     "Yes, concerns include patient privacy, data security, and the risk of AI bias affecting treatment decisions."] ,

    # 10 Examples with 4 turns
    ["How can the economy be impacted by rising inflation?", 
     "Rising inflation can erode purchasing power, reduce consumer spending, and lead to higher interest rates, which can slow economic growth.", 
     "What can be done to control inflation?", 
     "Central banks may raise interest rates or use other monetary policies to control inflation and stabilize the economy.", 
     "What are the risks of raising interest rates too high?", 
     "It can lead to a slowdown in investment and higher unemployment, as borrowing becomes more expensive."] ,

    ["How do mobile apps collect and use data?", 
     "Mobile apps collect data through permissions granted by users, including location, device information, and usage patterns, often used for advertising or improving user experience.", 
     "Is it safe to use apps that collect data?", 
     "It depends on the app’s privacy policy. Always review permissions and restrict unnecessary data access to ensure privacy.", 
     "How can you manage app permissions?", 
     "You can modify app permissions in your device settings, limiting access to sensitive data like location or contacts."] ,

    ["How can you stay productive when working from home?", 
     "Setting clear goals, creating a designated workspace, and establishing a daily routine are key to maintaining productivity when working remotely.", 
     "What are the challenges of working from home?", 
     "Distractions, isolation, and lack of clear boundaries between work and personal life can be challenging while working remotely.", 
     "What are some strategies to overcome these challenges?", 
     "Use time-blocking techniques, schedule regular check-ins with colleagues, and ensure you take breaks to avoid burnout."] ,

    ["What is the difference between electric cars and traditional gas-powered vehicles?", 
     "Electric cars are powered by electric motors and batteries, while traditional vehicles use internal combustion engines powered by gasoline or diesel.", 
     "What are the environmental benefits of electric cars?", 
     "Electric cars produce fewer emissions and are more energy-efficient, contributing to cleaner air and reduced carbon footprint.", 
     "Are electric cars more expensive?", 
     "Electric cars typically have a higher upfront cost, but the total cost of ownership is often lower due to fewer maintenance and fuel expenses."] ,

    ["What does it mean to live sustainably?", 
     "Living sustainably means making choices that minimize environmental impact, such as reducing waste, conserving resources, and supporting eco-friendly practices.", 
     "How can individuals live more sustainably?", 
     "People can reduce their carbon footprint by driving less, consuming less plastic, and supporting businesses that prioritize sustainability.", 
     "What role does recycling play in sustainability?", 
     "Recycling reduces waste, conserves natural resources, and lowers energy consumption, making it a key component of a sustainable lifestyle."] ,

    ["What are some common habits that can improve mental health?", 
     "Regular exercise, a balanced diet, social connections, and practicing mindfulness can all contribute to better mental well-being.", 
     "How do you know if you need professional help for mental health?", 
     "Persistent feelings of sadness, anxiety, or a lack of interest in daily activities are signs that it may be time to seek professional support.", 
     "What are the benefits of seeking help?", 
     "Professional support can provide strategies to cope with emotional challenges, reduce symptoms, and improve overall mental health."] ,

    ["How does the Internet of Things (IoT) work?", 
     "IoT connects everyday devices to the internet, allowing them to collect and share data, improving automation and decision-making processes.", 
     "What are some examples of IoT in daily life?", 
     "Smart thermostats, wearable health trackers, and connected home security systems are common IoT devices used in daily life.", 
     "What are the security risks of IoT devices?", 
     "IoT devices can be vulnerable to hacking if not properly secured, leading to data breaches and privacy concerns."] ,

    ["How do you define success?", 
     "Success can be defined in many ways: achieving personal goals, finding happiness, and maintaining fulfilling relationships, but it varies from person to person.", 
     "What factors contribute to long-term success?", 
     "Hard work, perseverance, adaptability, and the ability to learn from failures are essential for sustained success.", 
     "How do setbacks affect success?", 
     "Setbacks are inevitable, but they provide valuable learning opportunities that can lead to greater resilience and eventual success."] ,

    ["How can the fashion industry reduce waste?", 
     "The fashion industry can reduce waste through sustainable manufacturing, recycling materials, and encouraging consumers to buy less and reuse more.", 
     "How can consumers contribute to sustainability?", 
     "Consumers can choose brands that prioritize sustainability, buy second-hand, and recycle or upcycle old clothing.", 
     "Is fast fashion harmful to the environment?", 
     "Yes, fast fashion leads to high levels of waste, pollution, and exploitation of resources, contributing to environmental degradation."] ,

    ["How does global warming affect our daily lives?", 
     "Global warming leads to extreme weather events, rising sea levels, and disruptions in agriculture, all of which affect daily life.", 
     "What are some immediate impacts of global warming?", 
     "Heatwaves, flooding, and changes in food production are immediate effects, impacting health, food security, and living conditions.", 
     "What can individuals do to combat global warming?", 
     "Reducing energy consumption, using renewable energy, and advocating for policy changes are effective actions individuals can take."] ,

    # 10 Examples with 5 turns
    ["What are the benefits of a plant-based diet?", 
     "A plant-based diet is rich in nutrients, can lower the risk of chronic diseases, and has a smaller environmental footprint.", 
     "Are there any risks to a plant-based diet?", 
     "If not properly balanced, a plant-based diet may lack certain nutrients like vitamin B12, iron, and omega-3 fatty acids.", 
     "How can someone ensure they get all the nutrients?", 
     "By carefully planning meals and incorporating a variety of plant-based foods like beans, nuts, seeds, and fortified foods.", 
     "Is a plant-based diet suitable for everyone?", 
     "Yes, with proper planning, a plant-based diet can be tailored to meet the nutritional needs of people at all stages of life."] ,

    ["How does a credit score work?", 
     "A credit score is a numerical representation of a person's creditworthiness, based on factors like payment history, debt level, and credit history length.", 
     "What can lower your credit score?", 
     "Late payments, high credit utilization, and applying for too many new accounts can negatively impact your credit score.", 
     "How can you improve your credit score?", 
     "Paying bills on time, reducing outstanding debt, and avoiding unnecessary credit inquiries can help boost your score.", 
     "What is a good credit score?", 
     "A score of 700 or higher is typically considered good, but the range can vary depending on the credit scoring model."] ,

    ["What are the most common causes of stress?", 
     "Stress can be caused by factors like work pressure, financial problems, relationship issues, and health concerns.", 
     "How does stress affect your health?", 
     "Chronic stress can lead to heart disease, digestive problems, anxiety, and weakened immunity.", 
     "How can you manage stress effectively?", 
     "Exercise, meditation, time management, and seeking support from loved ones are effective ways to reduce stress.", 
     "What role does sleep play in stress management?", 
     "Quality sleep is essential for restoring energy, regulating mood, and reducing the physiological impacts of stress."] ,

    ["What is artificial intelligence?", 
     "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.", 
     "How is AI used in everyday life?", 
     "AI powers personal assistants like Siri, autonomous vehicles, recommendation systems, and even medical diagnostics.", 
     "Are there risks associated with AI?", 
     "AI could lead to job displacement, bias in decision-making, and ethical concerns related to privacy and security.", 
     "How can we ensure AI is used responsibly?", 
     "Developing ethical guidelines, promoting transparency, and ensuring AI systems are designed with fairness and accountability in mind are crucial."] ,

    ["What is a healthy work-life balance?", 
     "A healthy work-life balance means maintaining an equilibrium between professional responsibilities and personal well-being, including time for relaxation and family.", 
     "What are the consequences of poor work-life balance?", 
     "Burnout, increased stress, and decreased productivity are common outcomes of an unhealthy work-life balance.", 
     "How can you achieve a better work-life balance?", 
     "Setting boundaries, prioritizing tasks, and taking regular breaks can help achieve a more balanced lifestyle.", 
     "Why is work-life balance important for long-term success?", 
     "A healthy work-life balance improves overall well-being, boosts creativity, and helps sustain motivation over time."] ,

    ["What is the Internet of Things (IoT)?", 
     "The Internet of Things refers to a network of interconnected devices that can collect and exchange data without human intervention.", 
     "How does IoT benefit industries?", 
     "IoT enables automation, efficiency, and data-driven decision-making in sectors like manufacturing, healthcare, and agriculture.", 
     "What are the risks of IoT?", 
     "The main risks include privacy issues, security vulnerabilities, and potential over-reliance on automated systems.", 
     "How can IoT security be improved?", 
     "IoT devices should be secured with strong encryption, regular updates, and proper authentication protocols to minimize security risks."] ,

    ["What is the role of technology in education?", 
     "Technology enhances learning by providing access to information, facilitating collaboration, and supporting personalized education through tools like online platforms and digital resources.", 
     "How does technology support remote learning?", 
     "Technology enables communication, sharing resources, and interactive learning even when students are not physically present in the classroom.", 
     "Can technology replace traditional education?", 
     "While technology can enhance education, it is not a replacement for traditional teaching, which provides critical social interaction and hands-on experiences.", 
     "What are the future trends in educational technology?", 
     "Artificial intelligence, virtual reality, and adaptive learning systems are expected to further revolutionize education in the coming years."] 
]

def format_conversation(conversation):
    formatted_text = "<|begin_of_text|>"

    for message in conversation:
        role = message["role"]
        content = message["content"]
        formatted_text += f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"

    formatted_text += "<|end_of_text|>"  # Mark end of conversation
    return formatted_text
 
def process_hf_dataset():
    last_ = []
    print("\nDownloading dataset...\n")
    dataset = load_dataset("GAIR/lima")
    dataset["train"]["conversations"]
    print("Making `user` and `assistant` roles for each conversation...")
    for example in dataset["train"]["conversations"]:
        structured_data = []
        for i, turn in enumerate(example):
            role = "user" if i % 2 == 0 else "assistant"
            structured_data.append({"role": role, "content": turn})
        last_.append(format_conversation(structured_data))
    
    for example in gpt4o_generated_extra_turns:
        structured_data = []
        for i, turn in enumerate(example):
            role = "user" if i % 2 == 0 else "assistant"
            structured_data.append({"role": role, "content": turn})
        last_.append(format_conversation(structured_data))


    with open("dataset_LIMA/train_with_synthetic.jsonl", "w+", encoding="utf-8") as f:
        for line in last_:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
    print("Done")

if __name__ == "__main__":
    os.makedirs("dataset_LIMA", exist_ok=True)
    process_hf_dataset()

    