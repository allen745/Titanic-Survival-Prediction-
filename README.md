# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a passenger survived the **Titanic disaster** using Machine Learning.

The project follows a complete Machine Learning pipeline:

```
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Train Test Split
        ↓
Logistic Regression Model
        ↓
Model Evaluation
```

---

# 🎯 Objective

Build a classification model to predict passenger survival:

```
0 → Not Survived
1 → Survived
```

---

# 🛠️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 📂 Project Structure

```
Titanic-Survival-Prediction/

│
├── Titanic Survival Prediction.py
│
├── train.csv
├── test.csv
├── gender_submission.csv
│
├── count vs survival.png
├── count vs gender.png
├── count vs gender survival.png
├── count vs pclass.png
├── count vs Pclass survival.png
│
└── README.md

```

---

# 📊 Dataset Information

Dataset:

**Titanic Dataset (Kaggle)**


Dataset contains:

```
891 Rows
12 Columns
```


Features:

| Feature | Description |
|-|-|
| PassengerId | Passenger ID |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Gender |
| Age | Age |
| SibSp | Siblings/Spouse |
| Parch | Parents/Children |
| Ticket | Ticket number |
| Fare | Ticket price |
| Cabin | Cabin number |
| Embarked | Boarding port |
| Survived | Target variable |

---

# 🔎 Data Preprocessing

### Missing Value Treatment

Removed:

```
Cabin Column
```

Filled missing values:

```
Age → Mean Value

Embarked → Mode Value

Fare → Median Value
```

---

# 📈 Exploratory Data Analysis


## Survival Count

![Survival Count](count%20vs%20survival.png)


---

## Gender Distribution

![Gender Count](count%20vs%20gender.png)


---

## Gender Based Survival Analysis

![Gender Survival](count%20vs%20gender%20survival.png)


---

## Passenger Class Distribution

![Passenger Class](count%20vs%20pclass.png)


---

## Passenger Class vs Survival

![Class Survival](count%20vs%20Pclass%20survival.png)


---

# 🔄 Feature Encoding

Categorical values converted into numerical format:


### Sex Encoding

```
Male → 0

Female → 1
```


### Embarked Encoding

```
S → 0

C → 1

Q → 2
```

---

# 🤖 Machine Learning Model


## Logistic Regression

Used because:

- Binary classification problem
- Simple and efficient algorithm
- Good baseline model


---

# 🧠 Model Training


Features used:

```
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
```


Data Split:

```
Training Data : 80%

Testing Data  : 20%
```

---

# 📊 Model Evaluation


## Training Accuracy

```
80.75%
```


## Testing Accuracy

```
78.21%
```

---

# ✅ Result

The Logistic Regression model successfully predicts Titanic passenger survival with:

```
Test Accuracy ≈ 78%
```

---

# 🚀 How to Run Project


Clone repository:

```bash
git clone <your-repository-link>
```


Install dependencies:

```bash
pip install -r requirements.txt
```


Run:

```bash
python "Titanic Survival Prediction.py"
```

---

# 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# 👨‍💻 Author

**Allen Christian**

Machine Learning Project 🚀
