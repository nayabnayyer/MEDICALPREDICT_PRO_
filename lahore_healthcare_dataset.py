import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize faker and seed
fake = Faker()
np.random.seed(42)
random.seed(42)

# Define Lahore regions
lahore_regions = ['Johar Town', 'Gulberg', 'DHA', 'Model Town', 'Wapda Town',
                  'Iqbal Town', 'Samanabad', 'Cantt', 'Garden Town', 'Faisal Town']

# Function to generate realistic patient data for Lahore
def generate_lahore_patient_data(n=1500):
    data = []
    for _ in range(n):
        age = np.random.randint(18, 70)
        sex = random.choice(['male', 'female'])
        bmi = round(np.random.normal(24 if sex == 'female' else 26, 3.5), 1)
        children = np.random.randint(0, 5)
        smoker = np.random.choice(['yes', 'no'], p=[0.18, 0.82])
        region = random.choice(lahore_regions)

        # Area-based multiplier (higher for upscale areas)
        area_multiplier = {
            'DHA': 1.25,
            'Gulberg': 1.2,
            'Model Town': 1.1,
            'Johar Town': 1.05,
            'Garden Town': 1.0,
            'Wapda Town': 0.95,
            'Iqbal Town': 0.9,
            'Faisal Town': 0.9,
            'Samanabad': 0.85,
            'Cantt': 1.15
        }

        # Charges calculation
        base_cost = 3000
        age_factor = age * 28
        bmi_factor = (bmi - 21) * 120
        smoker_charge = 9000 if smoker == 'yes' else 0
        children_deduction = children * 200
        region_multiplier = area_multiplier.get(region, 1.0)

        charges = (base_cost + age_factor + bmi_factor + smoker_charge - children_deduction) * region_multiplier
        charges = round(max(charges, 2500), 2)  # ensure realistic minimum cost

        data.append({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': charges
        })

    return pd.DataFrame(data)

# Generate and save the Lahore data
df = generate_lahore_patient_data(1500)
df.to_csv('lahore_healthcare_data.csv', index=False)

print("Dataset generated and saved as 'lahore_healthcare_data.csv'")
