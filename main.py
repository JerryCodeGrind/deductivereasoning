from openai import OpenAI
import os
import json

from cases import cases

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def probabilistic_inference(doctor_vignette):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are an expert medical diagnosis assistant. Based on the patient information provided below, determine the top 20 most likely diagnoses, ordered from most to least likely.

Patient Information: {doctor_vignette}

Instructions:
1. Return EXACTLY 10 diseases, ordered from most to least likely
2. Each disease should be on its own line
3. Use the FULL, proper medical name for each diagnosis (e.g., "Heart Failure" not "Failure", "Chronic Obstructive Pulmonary Disease" not "COPD")
4. Include both common and rare diagnoses that fit the symptom profile
5. Do not include any additional text, explanations, or formatting
6. Do not include probabilities or other metadata

Example format:
Heart Failure
Chronic Obstructive Pulmonary Disease
Pneumonia
...etc."""}
        ]
    )
    answer = response.choices[0].message.content
    diseases = [d.strip() for d in answer.split('\n') if d.strip()]
    return diseases

def deductive_inference(doctor_vignette, least_likely_disease):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a medical expert using differential diagnosis.
    
Patient Information: {doctor_vignette}

Disease to rule out: {least_likely_disease}

Generate a single, focused question that would help rule out this disease. Since this is the least likely diagnosis, focus on finding a critical symptom or sign that, if present, would strongly suggest this disease is NOT the correct diagnosis.

Guidelines:
1. Focus on finding a critical symptom or sign that would definitively rule out this disease
2. Questions should be direct and answerable with simple yes/no responses
3. Refer to the patient as "you" not "the patient"
4. Make the question specific to the disease being considered
5. Return ONLY the question, with no additional text, numbering, or formatting

Example for a rare disease like "Leprosy":
"Have you noticed any loss of sensation in your skin patches?"

Example for "Acute Myeloid Leukemia":
"Have you had any unusual bleeding or bruising recently?"""}
        ]
    )
    answer = response.choices[0].message.content
    return answer.strip()

def main():
    results = []
    total_questions = 0
    
    for case_idx, case in enumerate(cases, 1):
        case_result = {
            "case_number": case_idx,
            "doctor_vignette": case['doctor_vignette'],
            "diseases": [],
            "least_likely_disease": "",
            "ruling_out_question": ""
        }
        
        print(f"\nProcessing Case {case_idx}...")
        doctor_vignette = case['doctor_vignette']
        diseases = probabilistic_inference(doctor_vignette)
        
        # Store all diseases
        case_result["diseases"] = diseases
        
        # Get the least likely disease (last in the list)
        least_likely_disease = diseases[-1]
        case_result["least_likely_disease"] = least_likely_disease
        
        # Generate one question to rule out the least likely disease
        question = deductive_inference(doctor_vignette, least_likely_disease)
        case_result["ruling_out_question"] = question
        total_questions += 1
        
        print(f"\nAll Diseases (most to least likely):")
        for i, disease in enumerate(diseases, 1):
            print(f"{i}. {disease}")
        print(f"\nLeast Likely Disease: {least_likely_disease}")
        print(f"Ruling Out Question: {question}")
        
        results.append(case_result)
    
    # Save results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal cases processed: {len(cases)}")
    print(f"Total questions generated: {total_questions}")
    print("\nResults have been saved to results.json")

if __name__ == "__main__":
    main()