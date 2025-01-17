# EthioMart Amharic NER Challenge

## **Project Overview**
This project supports EthioMart's goal of centralizing Telegram-based e-commerce activities in Ethiopia. The focus is on fine-tuning an **Amharic Named Entity Recognition (NER)** system to extract key entities such as:
- **Product Names**
- **Prices**
- **Locations**

Extracted entities will populate EthioMart's centralized database for streamlined e-commerce.

---

## **Objectives**
1. Extract and preprocess Amharic text data from Telegram.
2. Label datasets in CoNLL format for NER training.
3. Fine-tune and evaluate NER models.
4. Compare model performance.
5. Use interpretability tools (SHAP, LIME) to explain results.

---

## **Workflow**

### 1. **Data Collection**
- Use Telegram API (Telethon) to scrape data.
- Preprocess and structure text data for analysis.

### 2. **Data Labeling**
- Annotate 30-50 messages in CoNLL format:
  ```
  ዋጋ    B-PRICE
  1000   I-PRICE
  ብር    I-PRICE
  ```

### 3. **Model Fine-Tuning**
- Use pre-trained models like XLM-Roberta or AfroXMLR.
- Train using labeled data and evaluate with F1-score, precision, and recall.

### 4. **Model Comparison**
- Compare accuracy, speed, and robustness.
- Select the best-performing model.

### 5. **Model Interpretability**
- Use SHAP and LIME for insights and improvements.

---

## **Usage**

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/Serkalem-negusse1/Amharic-NER.git
   cd Amharic-NER
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Telegram API in `config.ini`:



### Commands:
- Extract data:
  ```bash
  python data_extraction.py --channel "@aradabrand2"
  ```
- Fine-tune models:
  ```bash
  python fine_tune_ner.py --model "xlm-roberta" --epochs 5
  ```
- Evaluate models:
  ```bash
  python evaluate_models.py --results_dir "results/"
  ```






