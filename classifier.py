#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from transformers import pipeline
import torch

companies_df = pd.read_csv("ml_insurance_challenge.csv")
taxonomy_df = pd.read_excel("insurance_taxonomy.xlsx")

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

def create_input_text(row):
    return f"{row['description']} Tags: {row['business_tags']} Sector: {row['sector']}, Category: {row['category']}, Niche: {row['niche']}"

companies_df["input_text"] = companies_df.apply(create_input_text, axis=1)

candidate_labels = taxonomy_df["label"].dropna().tolist()

def classify_text(text):
    result = classifier(text, candidate_labels, multi_label=True)
    top_labels = result["labels"][:3] 
    return ", ".join(top_labels)

print("Classifying companies... (this may take a while)")
companies_df["insurance_label"] = companies_df["input_text"].apply(classify_text)


companies_df[["description", "business_tags", "sector", "category", "niche", "insurance_label"]].to_csv("classified_companies.csv", index=False)

print("Fișierul 'classified_companies.csv' a fost salvat.")


#Am folosit Zero-shot modelul (facebook/bart-large-mnli) si poate clasifica fără date etichetate. 
#Fiecare clasificare a fost comparată semantic cu descrierea, tag-urile și contextul firmei.
#Puncte forte:
#Unul din punctele forte este ca modelul intelege sensul propozitiilor.
#Modelul este deja pre antrenat pe diverse tipuri de texte si este implicat in contextul de bussines, tech, agricultura.
#Nu necesita preantrenat.
#Puncte slabe:
#Timpul de executie este foarte lent pe CPU.

#Am lucrat in Colab de la  Google.
#Exemplu de output:
#-description
#Welchcivils is a civil engineering and construction company that specializes in designing and building utility network connections across the UK. They offer multi-utility solutions that combine electricity, gas, water, and fibre optic installation into a single contract. Their design engineer teams are capable of designing electricity, water and gas networks from existing network connection points to meter locations at the development, as well as project management of reinforcements and diversions. They provide custom connection solutions that take into account any existing assets, maximize the usage of every trench, and meet project deadlines. Welchcivils has considerable expertise installing gas and electricity connections in a variety of market categories, including residential, commercial, and industrial projects, as as well.
#-business_tags
#['Construction Services', 'Multi-utilities', 'Utility Network Connections Design and Construction', 'Water Connection Installation', 'Multi-utility Connections', 'Fiber Optic Installation']
#-sector
#Services
#-category
#Civil Engineering Services
#-niche
#Other Heavy and Civil Engineering Construction
#-insurance_label
#Commercial Construction Services, Gas Installation Services, SEO Services

