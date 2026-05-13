# 🤖 AI-Based Recommendation System with FastAPI

A production-style recommendation system API built with **FastAPI** and **scikit-learn**.  
Implements two complementary recommendation strategies:

| Strategy | Endpoint | Description |
|---|---|---|
| User-Based Collaborative Filtering | `GET /recommend` | Finds similar users → predicts items the target user will like |
| Item-Item Cosine Similarity | `GET /similar_items` | Finds items most similar to a query item |

---

## 📁 Project Structure

```
ai-recommendation-system-fastapi/
│
├── app/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI app + route definitions
│   ├── recommender.py       # Core ML logic (CF + cosine similarity)
│   ├── data_loader.py       # CSV loading + user-item matrix builder
│   └── utils.py             # Shared helper functions
│
├── data/
│   └── sample_data.csv      # 50-row user-item-rating dataset
│
├── notebooks/
│   └── exploration.ipynb    # EDA + visualisations
│
├── requirements.txt
├── README.md
├── .gitignore
└── run.sh
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-recommendation-system-fastapi.git
cd ai-recommendation-system-fastapi
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
# Using the convenience script
bash run.sh

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at **http://localhost:8000**  
Interactive docs (Swagger UI) at **http://localhost:8000/docs**

---

## 🔌 API Usage

### Health Check

```bash
GET /
```

```json
{"status": "ok", "service": "AI Recommendation System"}
```

---

### Get Recommendations for a User

```
GET /recommend?user_id=1
```

**cURL example:**

```bash
curl "http://localhost:8000/recommend?user_id=1"
```

**Response:**

```json
{
  "user_id": 1,
  "top_n": 5,
  "recommendations": [
    {"item_id": 104, "predicted_score": 3.9521},
    {"item_id": 106, "predicted_score": 3.7843},
    {"item_id": 108, "predicted_score": 3.5012},
    {"item_id": 109, "predicted_score": 3.2987},
    {"item_id": 110, "predicted_score": 3.1045}
  ]
}
```

**Error (invalid user):**

```bash
curl "http://localhost:8000/recommend?user_id=999"
```

```json
{"detail": "user_id=999 not found. Valid user IDs: [1, 2, 3, ...]"}
```

---

### Get Similar Items

```
GET /similar_items?item_id=101
```

**cURL example:**

```bash
curl "http://localhost:8000/similar_items?item_id=101"
```

**Response:**

```json
{
  "item_id": 101,
  "top_n": 5,
  "similar_items": [
    {"item_id": 107, "similarity_score": 0.9823},
    {"item_id": 103, "similarity_score": 0.9541},
    {"item_id": 104, "similarity_score": 0.9102},
    {"item_id": 106, "similarity_score": 0.8877},
    {"item_id": 105, "similarity_score": 0.8634}
  ]
}
```

---

## 🧠 How It Works

### User-Based Collaborative Filtering (`/recommend`)

1. Build a **user-item interaction matrix** from the ratings CSV.
2. **Mean-centre** each user's ratings to remove rating-scale bias.
3. Compute **cosine similarity** between all pairs of users.
4. For the target user, predict a score for each unrated item using a **weighted average** of neighbour ratings (weighted by similarity).
5. Return the top-5 highest-predicted items.

### Item-Item Cosine Similarity (`/similar_items`)

1. **Transpose** the user-item matrix so rows represent items.
2. Compute **cosine similarity** between all pairs of items.
3. Return the top-5 most similar items to the query item.

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| **FastAPI** | REST API framework |
| **uvicorn** | ASGI server |
| **pandas** | Data loading & manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | Cosine similarity computation |

---

## 📊 Extending the Dataset

Replace `data/sample_data.csv` with your own file following the same schema:

```
user_id,item_id,rating
1,101,4.5
1,102,3.0
...
```

No code changes are needed — the engine auto-adapts to the new matrix dimensions.

---

## 🧪 Running the Notebook

```bash
pip install jupyter matplotlib seaborn
jupyter notebook notebooks/exploration.ipynb
```

---

## 📄 License

MIT — free to use for personal and commercial projects.
