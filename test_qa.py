#!/usr/bin/env python
from crypto_qa import CryptoDueDiligenceQA

def test_qa_system():
    # Initialize QA system
    qa = CryptoDueDiligenceQA()
    
    # Test questions
    test_questions = [
        "What are the main legal concerns for crypto projects?",
        "How should I evaluate a crypto project's team?",
        "What technical security measures should a good crypto project have?",
        "What is the current market sentiment for Bitcoin?",
        "How do I assess the tokenomics of a new project?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = qa.answer_question(question)
        print(f"Answer: {answer}")
    
    qa.close()

if __name__ == "__main__":
    test_qa_system()