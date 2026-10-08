import pandas as pd
import random
import os

random.seed(42)

# ============================================================
# AI DOCUMENT RISK & COMPLIANCE CLASSIFIER
# DATASET GENERATOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(DATASET_DIR, "documents.csv")

os.makedirs(DATASET_DIR, exist_ok=True)


# ============================================================
# DOCUMENT DATA
# ============================================================

documents = {

    "Low": [
        "This document contains routine office procedures and general administrative guidelines.",
        "The document describes standard meeting procedures and internal communication practices.",
        "This document explains general employee onboarding procedures and office requirements.",
        "The document contains routine operational instructions for maintaining office equipment.",
        "This document describes standard business processes and general workflow guidelines.",
        "The document contains general information about company meetings, schedules, and responsibilities.",
        "This document describes routine purchasing procedures for office supplies.",
        "The document contains standard guidelines for organizing internal company events.",
        "This document explains normal administrative procedures followed by the organization.",
        "The document provides general instructions for employees regarding workplace procedures.",
        "This document describes standard travel request and reimbursement procedures.",
        "The document contains routine guidelines for submitting internal requests.",
        "This document explains standard document management and filing procedures.",
        "The document contains general information about office operations and responsibilities.",
        "This document describes routine communication and reporting procedures.",
        "The document provides standard guidelines for scheduling internal meetings.",
        "The document explains ordinary procedures for requesting company resources.",
        "This document contains general workplace guidelines and administrative instructions.",
        "The document describes standard processes for maintaining internal records.",
        "This document contains routine operational information without sensitive business data."
    ],

    "Medium": [
        "This agreement contains employee information and requires appropriate handling of internal records.",
        "The document includes customer information and specifies limited access for authorized employees.",
        "This policy describes data retention requirements for internal business records.",
        "The agreement contains confidential business information that should not be shared externally.",
        "This document includes vendor information and requires approval before external disclosure.",
        "The policy describes procedures for storing customer records and limiting employee access.",
        "This contract contains commercially sensitive information and confidentiality requirements.",
        "The document specifies retention periods for business records and internal communications.",
        "This agreement includes financial information that should only be accessed by authorized personnel.",
        "The document contains internal company information and requires controlled distribution.",
        "This policy requires employees to protect confidential customer and vendor information.",
        "The contract includes business terms that should not be disclosed without authorization.",
        "This document contains employee records that require controlled access and secure storage.",
        "The policy defines procedures for handling confidential operational information.",
        "This agreement includes internal pricing information and restrictions on disclosure.",
        "The document requires secure storage of customer information and periodic access reviews.",
        "This contract contains sensitive commercial terms and confidentiality obligations.",
        "The document defines internal access controls for business information.",
        "This policy requires employees to follow data retention and information handling procedures.",
        "The agreement contains confidential information that must be protected from unauthorized access."
    ],

    "High": [
        "This document contains personally identifiable information and requires strict privacy controls.",
        "The agreement involves customer financial information and requires compliance with privacy regulations.",
        "This policy governs the handling of sensitive personal data and requires restricted access.",
        "The contract permits third-party access to customer information and requires security controls.",
        "This document contains confidential financial records and requires regulatory compliance.",
        "The agreement includes personal data, financial information, and strict data retention obligations.",
        "This policy describes processing of sensitive customer information and mandatory security measures.",
        "The document requires encryption and access controls for confidential customer records.",
        "This contract contains sensitive information that may create significant regulatory obligations.",
        "The policy requires reporting of privacy incidents involving customer information.",
        "This agreement permits processing of personal information by an external service provider.",
        "The document contains sensitive financial information and requires compliance monitoring.",
        "This policy specifies strict controls for transferring customer data to third parties.",
        "The contract includes confidentiality, privacy, and security requirements for sensitive information.",
        "This document requires secure handling of personally identifiable customer information.",
        "The agreement includes regulatory reporting obligations related to customer data.",
        "This policy establishes strict controls for access to sensitive financial records.",
        "The document contains personal information and requires documented authorization for access.",
        "This contract involves third-party processing of sensitive customer information.",
        "The policy requires security reviews and incident reporting for protected business information."
    ],

    "Critical": [
        "This document contains highly sensitive personal and financial information subject to strict regulatory requirements.",
        "The agreement involves large volumes of customer data and significant privacy and security obligations.",
        "This contract permits processing of sensitive personal information and requires strict regulatory compliance.",
        "The document contains confidential financial records and creates substantial legal and regulatory exposure.",
        "This policy governs highly sensitive customer information and requires immediate compliance monitoring.",
        "The agreement includes sensitive personal data, financial records, and third-party processing obligations.",
        "This document involves regulated information and requires strict access controls, encryption, and audit logging.",
        "The contract contains sensitive financial and personal information with significant regulatory penalties for violations.",
        "This policy requires immediate reporting of serious privacy incidents involving protected customer information.",
        "The document permits international transfer of sensitive personal information subject to regulatory restrictions.",
        "This agreement involves highly confidential financial information and strict legal reporting requirements.",
        "The policy governs sensitive regulated data and requires continuous security monitoring.",
        "This contract includes extensive customer data processing and significant compliance obligations.",
        "The document contains highly sensitive information that could create severe regulatory consequences if disclosed.",
        "This agreement involves regulated financial information and mandatory incident notification requirements.",
        "The policy establishes strict controls for highly sensitive personal and financial records.",
        "This document includes sensitive information subject to legal, privacy, security, and regulatory requirements.",
        "The contract permits access to highly sensitive customer records and requires extensive security controls.",
        "This policy involves serious compliance obligations related to personal and financial information.",
        "The document contains regulated information where unauthorized disclosure could result in substantial penalties."
    ]
}


# ============================================================
# POLICY CATEGORIES
# ============================================================

policy_categories = {
    "Low": [
        "General Business",
        "Operations",
        "Administration"
    ],

    "Medium": [
        "Confidentiality",
        "Data Handling",
        "Internal Policy"
    ],

    "High": [
        "Privacy",
        "Financial Compliance",
        "Data Security"
    ],

    "Critical": [
        "Regulatory Compliance",
        "Privacy",
        "Financial Compliance",
        "Data Security"
    ]
}


# ============================================================
# GENERATE DATASET
# ============================================================

rows = []

DOCUMENTS_PER_RISK = 80

for risk_level, templates in documents.items():

    for i in range(DOCUMENTS_PER_RISK):

        text = random.choice(templates)

        variations = [
            " The requirements must be followed by authorized personnel.",
            " Access should be restricted to approved employees.",
            " The organization must maintain appropriate records.",
            " Employees must follow the applicable company policy.",
            " The information should be handled according to company procedures.",
            " Appropriate controls should be maintained throughout the process."
        ]

        text += random.choice(variations)

        policy = random.choice(
            policy_categories[risk_level]
        )

        rows.append({
            "document_id": f"DOC-{len(rows) + 1:04d}",
            "text": text,
            "risk_level": risk_level,
            "policy_category": policy
        })


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(rows)

df = pd.DataFrame(rows)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("AI DOCUMENT RISK & COMPLIANCE CLASSIFIER")
print("DATASET GENERATION")
print("=" * 60)

print(f"\nDataset saved to:")
print(OUTPUT_FILE)

print(f"\nTotal documents: {len(df)}")

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

print("\nPolicy distribution:")
print(df["policy_category"].value_counts())

print("\nDataset preview:")
print(df.head())

print("\nColumns:")
print(list(df.columns))

print("\nDataset creation completed successfully.")

print("=" * 60)