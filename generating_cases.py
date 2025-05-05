from openai import OpenAI
import json
import time
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")

def validate_case(case):
    # Ensure the case is a dict with the required fields and non-empty values
    if not isinstance(case, dict):
        return False
    if "doctor_vignette" not in case or "actual_diagnosis" not in case:
        return False
    if not case["doctor_vignette"].strip() or not case["actual_diagnosis"].strip():
        return False
    return True

def parse_case_response(response_content):
    # Try to parse as JSON, fallback to extracting fields manually
    try:
        case = json.loads(response_content)
        if validate_case(case):
            return case
    except Exception:
        pass
    # Fallback: try to extract fields manually
    try:
        lines = response_content.splitlines()
        vignette = ""
        diagnosis = ""
        for line in lines:
            if 'doctor_vignette' in line:
                vignette = line.split(':', 1)[-1].strip(' ",')
            if 'actual_diagnosis' in line:
                diagnosis = line.split(':', 1)[-1].strip(' ",')
        if vignette and diagnosis:
            return {"doctor_vignette": vignette, "actual_diagnosis": diagnosis}
    except Exception:
        pass
    return None

def generate_unique_diseases(num_diseases: int, max_retries: int = 3) -> List[str]:
    """Generate a list of unique medical diagnoses."""
    diseases = set()
    prompt = """Generate a list of unique medical diagnoses. Each diagnosis should be:
1. A specific disease or condition (not a symptom or general category)
2. In proper medical terminology
3. Capitalized
4. Not an abbreviation
5. Different from these examples: Paraneoplastic Hypercalcemia, Pemphigus Vulgaris, Tubo-ovarian Abscess, Leprosy, Goodpasture Syndrome

Return ONLY the diagnoses, one per line, with no additional text or formatting."""

    while len(diseases) < num_diseases:
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical diagnosis generator. Generate unique medical diagnoses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            content = response.choices[0].message.content
            new_diseases = [d.strip() for d in content.split('\n') if d.strip()]
            diseases.update(new_diseases)
            print(f"Generated {len(diseases)} unique diseases so far...")
        except Exception as e:
            print(f"Error generating diseases: {e}")
            time.sleep(1)
    
    return list(diseases)[:num_diseases]

def generate_vignette_for_disease(disease: str, max_retries: int = 3) -> Dict:
    """Generate a doctor vignette for a specific disease."""
    prompt = f"""Generate a realistic doctor vignette for the following diagnosis: {disease}

OUTPUT FORMAT (JSON):
{{
  "doctor_vignette": [
    "1. Demographics (age, sex, relevant history)",
    "2. Chief complaint + symptom details (duration, severity)",
    "3. Key physical exam findings (vitals, notable signs)",
    "4. Critical lab/imaging result (if applicable)"
  ].join(' '),
  "actual_diagnosis": "{disease}"
}}

CONTENT RULES:
- Vignette: 3-4 sentences max, no paragraph breaks
- Each component of the vignette must be present and in order
- Use proper medical terminology
- Make the case realistic and clinically relevant
- Focus on the most common presentation of the disease"""

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical case generator. Follow the format and rules exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            content = response.choices[0].message.content
            case = parse_case_response(content)
            if validate_case(case):
                return case
        except Exception as e:
            print(f"Error generating vignette for {disease} (attempt {attempt+1}): {e}")
        time.sleep(1)
    # Fallback if all attempts fail
    return {"doctor_vignette": f"Unknown vignette for {disease}.", "actual_diagnosis": disease}

def save_cases(cases, filename="medical_cases.json"):
    """Save cases to JSON file with pretty printing"""
    with open(filename, 'w') as f:
        json.dump(cases, f, indent=2)

def load_cases(filename="medical_cases.json"):
    """Load existing cases from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def generate_cases_parallel(num_cases=1000, max_workers=5):
    """Generate cases in parallel using ThreadPoolExecutor"""
    # Load existing cases if any
    cases = load_cases()
    start_idx = len(cases)
    
    if start_idx >= num_cases:
        print(f"Already have {start_idx} cases. No new cases needed.")
        return
    
    print(f"Found {start_idx} existing cases. Generating {num_cases - start_idx} new cases...")
    
    # First generate unique diseases
    print("\nGenerating unique diseases...")
    diseases = generate_unique_diseases(num_cases - start_idx)
    print(f"Generated {len(diseases)} unique diseases.")
    
    # Generate vignettes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_disease = {
            executor.submit(generate_vignette_for_disease, disease): disease 
            for disease in diseases
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_disease)):
            disease = future_to_disease[future]
            try:
                case = future.result()
                cases.append(case)
                
                # Save after each case
                save_cases(cases)
                
                # Print progress
                print(f"\nGenerated case {start_idx + i + 1}/{num_cases}:")
                print(f"Diagnosis: {case['actual_diagnosis']}")
                print(f"Vignette: {case['doctor_vignette']}")
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing case for {disease}: {e}")
    
    print(f"\nGeneration complete! Total cases: {len(cases)}")
    print("Results have been saved to 'medical_cases.json'")

if __name__ == "__main__":
    generate_cases_parallel(num_cases=1000, max_workers=5)