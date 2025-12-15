# **Food Inflation Data Insights**

An AI-powered web application for analyzing food price inflation data using natural language queries. Ask questions about inflation trends, outliers, top values, and country clusters — and get instant visual insights.

**Dataset Source:** [Kaggle — Food Price Inflation](https://www.kaggle.com/datasets/umitka/food-price-inflation/data)

---

## **Overview**

This project combines a **Vue.js frontend** with a **FastAPI backend** to deliver intelligent data analysis. Users can ask natural language questions about food inflation, and the system automatically detects the intent (outliers, trends, clustering, or top values) using sentence embeddings, then executes the appropriate analysis and returns visual results.

---

# **Frontend**

A modern Vue.js single-page application that provides an intuitive interface for querying food inflation data.

---

## **Technologies Used**

| Technology | Purpose                                    |
| ---------- | ------------------------------------------ |
| **Vue 3**  | Reactive UI framework with Composition API |
| **Vite**   | Fast build tool and development server     |
| **ESLint** | Code quality and linting                   |

---

## **Vue Page Overview**

The frontend consists of a single `App.vue` component that handles all user interactions and data visualization.

### **Main Sections**

| Section                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| **Header**               | Application title and subtitle                   |
| **Query Input**          | Textarea for entering natural language questions |
| **Response Display**     | Dynamic visualization of analysis results        |
| **Conversation History** | Track of previous queries and responses          |

---

## **Possible Contents & Displays**

The interface dynamically renders different visualizations based on the detected intent:

### **Outlier Detection**

Displays anomalous data points detected using Isolation Forest algorithm.

**Visual elements:**

- Grid of outlier cards
- Country name, time period, and inflation value
- Count badge showing total outliers found

---

### **Top Values Analysis**

Shows the highest inflation values in the dataset.

**Visual elements:**

- Ranked list with position badges (1st, 2nd, 3rd, etc.)
- Country name and time period
- Inflation percentage value

---

### **Country Clustering**

Groups countries by similar inflation behavior using K-Means clustering.

**Visual elements:**

- Cluster groups with color-coded headers
- List of countries per cluster
- Mean inflation value for each country

---

### **Trend Analysis**

Performs linear regression to identify overall inflation trends.

**Visual elements:**

- Trend direction indicator (increasing/decreasing)
- Slope and intercept values
- Data points count
- Simple bar chart showing predicted trend line

---

## **Response Metadata**

Each response includes:

- **Intent** — The detected analysis type
- **Confidence Score** — How confident the model is about the intent (High ≥80%, Medium ≥60%, Low <60%)
- **Color-coded Headers** — Each intent type has a unique gradient color scheme

---

## **Getting Started (Frontend)**

### **Prerequisites**

- Node.js (v20.19.0+ or v22.12.0+)

### **Running the Frontend**

1. **Navigate to the frontend directory**

   ```
   cd frontend
   ```

2. **Install dependencies**

   ```
   npm install
   ```

3. **Start the development server**

   ```
   npm run dev
   ```

4. **Open in browser** — The app runs at `http://localhost:5173` by default

---

## **Project Structure (Frontend)**

| Folder/File                     | Purpose                                      |
| ------------------------------- | -------------------------------------------- |
| `src/App.vue`                   | Main application component with all UI logic |
| `src/main.js`                   | Vue app entry point                          |
| `src/api/backend_connection.js` | API client for backend communication         |
| `public/`                       | Static assets                                |
| `vite.config.js`                | Vite configuration                           |
| `package.json`                  | Dependencies and scripts                     |

---

# **Backend**

A FastAPI-based REST API that processes natural language queries and performs data analysis using machine learning techniques.

---

## **Technologies Used**

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| **FastAPI**               | High-performance Python web framework |
| **Pydantic**              | Data validation and serialization     |
| **Sentence-Transformers** | Text embedding for intent detection   |
| **PyTorch**               | Tensor operations and model inference |
| **Pandas**                | Data manipulation and analysis        |
| **Scikit-learn**          | Machine learning algorithms           |

---

## **API Resources**

### **Query Endpoint**

The single endpoint that powers all analysis features.

**Endpoint:** `POST /query`

**Request body:**

```json
{
  "text": "What are the top inflation values?"
}
```

**Response structure:**

```json
{
  "intent": "top",
  "score": 0.85,
  "result": [...]
}
```

---

## **Analysis Capabilities**

### **Intent Detection**

Uses semantic similarity to classify user queries into one of four analysis types:

| Intent     | Trigger Keywords                              |
| ---------- | --------------------------------------------- |
| `outliers` | detect outliers, anomalies, isolation forest  |
| `top`      | top values, highest values, biggest inflation |
| `trend`    | trend, regression, line going up or down      |
| `cluster`  | cluster countries, similar inflation behavior |

---

### **Outlier Detection**

Identifies anomalous inflation values using Isolation Forest.

- **Algorithm:** Isolation Forest
- **Contamination:** 1%
- **Output:** DataFrame rows flagged as outliers

---

### **Top Values**

Returns the highest inflation observations.

- **Default count:** 10 records
- **Fields:** Country, Time Period, Observation Value

---

### **Country Clustering**

Groups countries by mean inflation using K-Means clustering.

- **Algorithm:** K-Means
- **Clusters:** 5 groups
- **Preprocessing:** StandardScaler normalization
- **Output:** Country, Mean Inflation, Cluster ID

---

### **Trend Regression**

Performs linear regression to identify inflation trends over time.

- **Algorithm:** Linear Regression
- **Output:** Slope, Intercept, Predicted values

---

## **Data Model**

The backend processes the following data structure:

```
food_price_inflation.csv
 ├── REF_AREA_LABEL (Country name)
 ├── TIME_PERIOD (Date/period)
 └── OBS_VALUE (Inflation value)
```

---

## **How It Connects to the Frontend**

| Frontend Feature | Backend Support                                 |
| ---------------- | ----------------------------------------------- |
| Query Input      | POST /query endpoint                            |
| Intent Display   | Intent detection with confidence score          |
| Outliers View    | Isolation Forest analysis                       |
| Top Values View  | nlargest() DataFrame operation                  |
| Cluster View     | K-Means clustering results                      |
| Trend View       | Linear regression slope, intercept, predictions |

---

## **Getting Started (Backend)**

### **Prerequisites**

- Python 3.10+
- pip

### **Running the API**

1. **Navigate to the backend directory**

   ```
   cd backend
   ```

2. **Create a virtual environment** (recommended)

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```
   pip install fastapi uvicorn pandas torch sentence-transformers scikit-learn
   ```

4. **Run the application**

   ```
   uvicorn app:app --reload
   ```

5. **Access the API** — The server runs at `http://127.0.0.1:8000`

6. **Explore the API** — Swagger UI available at `http://127.0.0.1:8000/docs`

---

## **Project Structure (Backend)**

| File                       | Purpose                                         |
| -------------------------- | ----------------------------------------------- |
| `app.py`                   | FastAPI application and endpoint definitions    |
| `model_utils.py`           | ML models, data loading, and analysis functions |
| `food_price_inflation.csv` | Source dataset                                  |
| `corpus_embeddings.pt`     | Cached sentence embeddings for faster inference |

---

## **Configuration Notes**

- **CORS** — Configured to allow all origins for development (`allow_origins=["*"]`)
- **Embeddings Cache** — Corpus embeddings are cached to `corpus_embeddings.pt` for faster startup
- **Model** — Uses `all-MiniLM-L6-v2` sentence transformer (runs on CPU)

---

## **Example Queries**

Try these natural language questions:

- "Show me the outliers in the data"
- "What are the top 10 highest inflation values?"
- "Cluster countries by their inflation patterns"
- "What's the overall trend in food prices?"

---
