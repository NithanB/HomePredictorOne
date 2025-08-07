# HomePredictorOne
HomePredictorOne is a machine learning project designed to predict home prices based on a variety of features such as location, size, age, and more. The goal is to provide accurate, easy-to-use interface predictions for homeowners, real estate agents, and buyers.

## Features
• Predicts home prices using advanced machine learning models.
• Supports multiple data sources and feature sets.
• Simple API for integrating with web or mobile application.
• Easy-to-use command-line interface for batch predictions.
• Visualization tools for analyzing prediction results

## Background
From the dataset given by Aj. Ekarat(https://github.com/ekaratnida), I have added some of the additional features from the given sample video.

I found that finding a relationship between house prices with number of rooms and area is quite ambiguous when houses are located differently in one city. For example, houses are expensive when it is located near to the city center when comparing to the further counterparts. Therefore, I am adding city zone numbers as the new table with maximum of 3 zones:

Given that you (user) are working as a real estate agent in Bangkok City and you are assigned to predict prices in different city zones

Zone 1 : Inside the MRT Blue Line Circle, adjacent to BTS green line 

Zone 2 : Outside the MRT Blue Line Circle

Zone 3 : Far West and Far East of Bangkok (Bangna, Nong Chok, Thonburi, Rama II, etc.)

The number is multiplied by 3 to increase the accuracy of the predictions

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package installer)
- (Optional) Jupyter Notebook for data exploration

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/HomePredictorOne.git
cd HomePredictorOne
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Command Line

```bash
python predict.py --input data/sample_homes.csv --output predictions.csv
```

#### As a Library

```python
from homepredictorone import HomePredictor

predictor = HomePredictor(model_path="models/latest.pkl")
result = predictor.predict({
    "location": "New York",
    "size_sqft": 1400,
    "bedrooms": 3,
    "age": 10
})
print(f"Predicted price: ${result}")
```

#### API

Run the API:

```bash
uvicorn api:app --reload
```

Then send a POST request to `/predict` with home features in the JSON body.

### Model Training

To train your own model:

```bash
python train.py --data data/training_set.csv --output models/new_model.pkl
```

## Project Structure

```
HomePredictorOne/
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── predictor.py
│   └── ...
├── api.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

