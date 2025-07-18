import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Initialize Faker and seed
fake = Faker()
np.random.seed(42)
random.seed(42)

# Lahore-specific regions
lahore_regions = ['Gulberg', 'Johar Town', 'DHA', 'Model Town', 'Wapda Town', 'Iqbal Town', 'Cantt']

# Generate synthetic patient data
def generate_patient_data(n=1500):
    data = []
    for _ in range(n):
        age = np.random.randint(18, 70)
        sex = random.choice(['male', 'female'])
        bmi = round(np.random.normal(24 if sex == 'female' else 26, 4), 1)
        children = np.random.randint(0, 6)
        smoker = np.random.choice(['yes', 'no'], p=[0.2, 0.8])
        region = random.choice(lahore_regions)

        # Estimate charges
        base_cost = 3000
        age_factor = age * 30
        bmi_factor = (bmi - 21) * 150
        smoker_charge = 8000 if smoker == 'yes' else 0
        children_deduction = children * 250
        region_factor = random.randint(-1500, 1500)

        charges = base_cost + age_factor + bmi_factor + smoker_charge - children_deduction + region_factor
        charges = round(max(charges, 2000), 2)

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

# Generate the data
df = generate_patient_data(1500)

# Save the dataset
output_path = 'C:/Users/DELL/Desktop/medical cost prediction/lahore_healthcare_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Dataset generated and saved at: {output_path}")

# Verify file saved successfully and preview
if os.path.exists(output_path):
    df_loaded = pd.read_csv(output_path)
    print("📂 File found! Here's a preview:")
    print(df_loaded.head())
else:
    print("❌ File not found at this path.")
