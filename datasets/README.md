# Datasets

This folder contains datasets used throughout the AI Red Teamer modules.

## SMS Spam Collection Dataset

**Location:** `sms_spam_collection/SMSSpamCollection`

**Description:** The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research. It contains 5,574 messages in English, tagged as either ham (legitimate) or spam.

**Source:** UCI Machine Learning Repository  
**Original Authors:**
- Tiago A. Almeida (Universidade Federal de Sao Carlos, Brazil)
- José María Gómez Hidalgo (R&D Department, Optenet, Spain)

**File Format:** Tab-separated values (TSV)
- Column 1: Label (ham or spam)
- Column 2: Message text

**Usage:** Used in Module 02 - Applications of AI in InfoSec for spam classification exercises.

**Citation:**
```
Almeida, T.A., Hidalgo, J.M.G. and Yamakami, A., 2011, September. 
Contributions to the study of SMS spam filtering: new collection and results. 
In Proceedings of the 11th ACM symposium on Document engineering (pp. 259-262).
```

## Adding New Datasets

When adding new datasets to this repository:
1. Create a subdirectory with a descriptive name
2. Include the raw dataset files
3. Add a section in this README describing the dataset
4. Include source, citation, and usage information
