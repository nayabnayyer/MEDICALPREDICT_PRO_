import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize faker and seed
fake = Faker()
np.random.seed(42)
random.seed(42)

# Define Pakistani regions
regions = ['Punjab', 'Sindh', 'KPK', 'Balochistan', 'Islamabad']

# Define function to generate dataset
def generate_patient_data(n=1500):
    data = []
    for _ in range(n):
        age = np.random.randint(18, 70)
        sex = random.choice(['male', 'female'])
        bmi = round(np.random.normal(24 if sex == 'female' else 26, 4), 1)
        children = np.random.randint(0, 6)
        smoker = np.random.choice(['yes', 'no'], p=[0.2, 0.8])  # smoking is less common
        region = random.choice(regions)

        # Charges logic (simplified)
        base_cost = 3000
        age_factor = age * 30
        bmi_factor = (bmi - 21) * 150
        smoker_charge = 8000 if smoker == 'yes' else 0
        children_deduction = children * 250
        region_factor = random.randint(-2000, 2000)

        charges = base_cost + age_factor + bmi_factor + smoker_charge - children_deduction + region_factor
        charges = round(max(charges, 2000), 2)  # ensure minimum cost

        data.append({
            'age': age,
            'sex': sex,
            'bmi': round(bmi, 1),
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': charges
        })

    return pd.DataFrame(data)

# Generate and save the data
df = generate_patient_data(1500)
df.to_csv('pakistan_healthcare_data.csv', index=False)

print("Dataset generated and saved as 'pakistan_healthcare_data.csv'")