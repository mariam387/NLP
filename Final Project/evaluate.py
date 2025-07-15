from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from sentence_transformers import SentenceTransformer, util

# Load models
llm = OllamaLLM(model="llama3.2")
evaluator = SentenceTransformer("all-MiniLM-L6-v2")

# Prompt template
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# Test set (your benchmark questions + expected answers)
test_data = [
    {
        "question": "What are the vegan options?",
        "expected": "The restaurant offers vegan pizza and salads."
    },
    {
        "question": "Do they offer delivery on weekends?",
        "expected": "Yes, delivery is available on weekends."
    },
    {
        "question": "Is the pizza spicy?",
        "expected": "The pizza is not very spicy, but you can request extra spice."
    },
    # ➕ Add more questions here
]

# Evaluation loop
def evaluate_answer(predicted, expected):
    emb_pred = evaluator.encode(predicted, convert_to_tensor=True)
    emb_exp = evaluator.encode(expected, convert_to_tensor=True)
    return util.cos_sim(emb_pred, emb_exp).item()

scores = []

for i, item in enumerate(test_data):
    question = item["question"]
    expected = item["expected"]

    reviews = retriever.invoke(question)
    prediction = chain.invoke({"reviews": reviews, "question": question})

    score = evaluate_answer(prediction, expected)
    scores.append(score)

    print(f"\n----- Q{i+1} -----")
    print("❓ Question:", question)
    print("🤖 Model Answer:", prediction)
    print("✅ Expected Answer:", expected)
    print(f"📊 Similarity Score: {score:.2f}")

# Final average
average_score = sum(scores) / len(scores)
print("\n========================")
print(f"🔎 Average Similarity Score: {average_score:.2f}")
print("========================")
