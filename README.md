# 📊 Marketing Campaign Analytics & Customer Journey Optimization

> Comprehensive analytics project analyzing 100,000 customers and $8.9M in e-commerce transaction data to optimize marketing campaigns, predict customer churn, and maximize ROI.

---

## 🎯 Project Overview

This project addresses critical marketing challenges through data-driven analysis:

**Key Questions Answered:**
- Which marketing channels deliver the highest ROI?
- How do multi-channel customers differ from single-channel customers?
- Which customers are at risk of churning?
- How can we optimize budget allocation across channels?
- What does the customer journey look like from first touch to purchase?

**Dataset:** E-Commerce Marketing & Analytics (Kaggle)  
**Tools:** Python, Pandas, Scikit-learn, Tableau  
**Timeline:** December 2024

---

## 🔑 Key Findings

### 💰 Business Impact

| Metric | Value | Insight |
|--------|-------|---------|
| **Total Revenue Analyzed** | $8.9M | Across 50 marketing campaigns |
| **Multi-Channel Uplift** | +104% | Multi-channel customers spend 2x more |
| **At-Risk Customer Value** | $1.67M | High-priority retention opportunity |
| **ROI Improvement** | +18% | Through optimized budget allocation |
| **Champions LTV** | $297 | 3.1x higher than average ($96) |

### 📈 Top Insights

1. **Multi-channel customers generate 104% higher revenue** ($243 vs $119 average)
2. **23.3% of customers engage across multiple channels** - opportunity to increase
3. **Champions segment (15.8%) drives outsized value** - 3.1x higher lifetime value
4. **14,996 customers at high risk of churn** - worth $1.67M in potential lost revenue
5. **Affiliate and Paid Search channels** drive 47% of total campaign revenue
6. **Budget optimization model** projects 18% ROI improvement through reallocation

---

## 📊 Dataset

### Source
E-Commerce Marketing & Analytics Dataset from Kaggle

### Structure

| File | Records | Description |
|------|---------|-------------|
| **customers.csv** | 100,000 | Customer demographics, loyalty tier, acquisition channel |
| **campaigns.csv** | 50 | Campaign details, channel, objective, dates |
| **products.csv** | 2,000 | Product catalog with categories and pricing |
| **transactions.csv** | 103,127 | Purchase data linking customers, campaigns, products |
| **events.csv** | 103,127 | Behavioral events (views, clicks, add-to-cart, purchases) |

### Data Quality
- Removed 10,449 transactions with missing product IDs
- Final dataset: 92,678 clean transactions
- Timeframe: January 2021 - December 2023

---

## 🔬 Methodology

### Phase 1: Data Preparation
```
1. Data loading and exploration
2. Missing value treatment
3. Feature engineering (date components, revenue calculations)
4. Dataset merging (customers → campaigns → products → transactions)
```

### Phase 2: Core Analytics

#### Campaign Performance Analysis
- Revenue and conversion metrics by campaign
- Channel effectiveness comparison
- Campaign objective analysis (Acquisition, Retention, Cross-sell)

#### Customer Segmentation (RFM)
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending
- **Segments**: Champions, Loyal, At-Risk, Lost, New, Potential

#### Multi-Touch Attribution
- First-touch attribution (acquisition)
- Last-touch attribution (conversion)
- Multi-channel journey analysis

#### Product Analytics
- Category performance analysis
- Premium vs Regular product comparison
- Top product identification

### Phase 3: Advanced Analytics

#### Churn Prediction Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 85.3%
- **Features**: Recency, frequency, monetary, demographics, segment
- **Output**: Churn probability score for each customer

#### Budget Optimization
- Current allocation analysis
- ROI-weighted optimal allocation
- Projected revenue impact

#### Conversion Funnel Analysis
- View → Click → Add to Cart → Purchase
- Channel-specific conversion rates
- Device performance analysis
- A/B test results

---

## 📈 Results & Insights

### 1. Campaign Performance

| Channel | Revenue | Transactions | Customers | Avg ROAS |
|---------|---------|--------------|-----------|----------|
| Affiliate | $1,697,655 | 17,676 | 16,179 | 4.0x |
| Paid Search | $1,629,720 | 16,949 | 15,585 | 4.0x |
| Email | $1,439,207 | 15,108 | 14,044 | 4.0x |
| Display | $1,285,120 | 13,437 | 12,579 | 4.0x |
| Social | $1,035,550 | 10,719 | 10,134 | 4.0x |

**Insight:** Affiliate and Paid Search drive 47% of campaign revenue

### 2. Customer Segmentation

| Segment | Count | % of Base | Avg LTV | Avg Recency (days) |
|---------|-------|-----------|---------|-------------------|
| Champions | 9,520 | 15.8% | $297 | 138 |
| Loyal Customers | 9,222 | 15.3% | $191 | 305 |
| At Risk | 11,686 | 19.4% | $147 | 765 |
| Lost | 12,331 | 20.5% | $96 | 808 |
| New Customers | 7,166 | 11.9% | $96 | 155 |
| Potential | 10,166 | 16.9% | $69 | 324 |

**Insight:** Focus retention on 11,686 "At Risk" customers worth $1.7M

### 3. Multi-Touch Attribution

- **Multi-channel customers**: 23.3% of base
- **Average spend**: $243 (multi-channel) vs $119 (single-channel)
- **Revenue uplift**: +104%
- **Average channels engaged**: 2.2

**Insight:** Coordinated multi-channel campaigns significantly increase customer value

### 4. Predictive Models

**Churn Prediction:**
- Model accuracy: 85.3%
- AUC-ROC: 0.85
- High-risk customers: 14,996
- At-risk revenue: $1.67M
- Top predictor: Recency (80.4% importance)

**Campaign Response:**
- Model accuracy: 78.2%
- Use case: Target high-probability responders
- Cost savings: Reduce wasted ad spend

### 5. Budget Optimization

**Projected Impact:**
- Current revenue: $7.1M
- Optimized revenue: $7.4M
- Increase: +$300K
- ROI improvement: +18%

**Recommended Changes:**
- Increase Email budget: +50%
- Increase Affiliate budget: +20%
- Decrease Social budget: -61%

### 6. Conversion Funnel

**Overall Funnel Performance:**
- View → Click: 15%
- Click → Add to Cart: 35%
- Add to Cart → Purchase: 25%
- **Overall conversion**: 1.3%

**Best Performing Channels:**
1. Email: 2.8% conversion
2. Paid Search: 1.9% conversion
3. Organic: 1.5% conversion

---

## 🛠️ Technologies Used

### Data Analysis
- **Python 3.8+**: Core programming
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models

### Visualization
- **Tableau**: Interactive dashboards
- **Matplotlib & Seaborn**: Statistical visualizations

### Tools
- **Jupyter Notebook**: Analysis documentation
- **Git & GitHub**: Version control
- **VS Code**: Development environment

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/marketing-analytics-project.git
cd marketing-analytics-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Download E-Commerce Marketing & Analytics Dataset from Kaggle
- Place CSV files in the project directory

### Running the Analysis

**Step 1: Base Analysis**
```bash
python marketing_analysis.py
```
Output: 6 CSV files with core metrics

**Step 2: Advanced Analytics**
```bash
python advanced_analytics.py
```
Output: Churn predictions, attribution analysis, budget optimization

**Step 3: Events & Funnel Analysis**
```bash
python events_analysis.py
```
Output: Conversion funnel, A/B test results, session metrics

**Step 4: View Results**
- Check generated CSV files
- Import `master_dataset_for_tableau.csv` into Tableau
- Build dashboards using the visualization guide

---

## 📁 Project Structure
```
marketing-analytics-project/
│
├── data/
│   ├── customers.csv
│   ├── campaigns.csv
│   ├── products.csv
│   ├── transactions.csv
│   └── events.csv
│
├── scripts/
│   ├── marketing_analysis.py          # Core analysis
│   ├── advanced_analytics.py          # Predictive models
│   └── events_analysis.py             # Funnel analysis
│
├── output/
│   ├── analysis_campaign_performance.csv
│   ├── analysis_customer_segments.csv
│   ├── analysis_attribution_fixed.csv
│   ├── analysis_churn_predictions_fixed.csv
│   ├── analysis_budget_optimization_fixed.csv
│   ├── analysis_channel_funnel.csv
│   ├── analysis_device_performance.csv
│   └── master_dataset_for_tableau.csv
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 📊 Interactive Dashboard

### Dashboard Components

The Tableau dashboard includes:

**Dashboard 1: Executive Overview**
- Total revenue, transactions, customers
- Channel performance comparison
- Monthly revenue trends
- Top campaigns

**Dashboard 2: Customer Segmentation**
- RFM segment distribution
- Segment characteristics
- Lifetime value analysis
- At-risk customer identification

**Dashboard 3: Campaign Performance**
- Campaign scatter plot (revenue vs customers)
- Channel comparison
- Objective effectiveness
- Multi-touch attribution

**Dashboard 4: Product Analytics**
- Category performance
- Premium vs Regular analysis
- Top products
- Product performance matrix

**Dashboard 5: Conversion Funnel**
- Stage-by-stage drop-off
- Channel-specific funnels
- Device performance
- A/B test results

---

## 💡 Key Recommendations

### Immediate Actions

1. **Launch Multi-Channel Campaigns**
   - Target single-channel customers with complementary channels
   - Potential: +104% revenue uplift across 46,000 customers

2. **At-Risk Customer Retention**
   - Create targeted campaigns for 11,686 at-risk customers
   - Potential: Recover $1.7M in revenue

3. **Budget Reallocation**
   - Increase Email marketing budget by 50%
   - Shift budget from Social to Affiliate channels
   - Projected: +18% ROI improvement

4. **Champions Program**
   - Develop VIP program for 9,520 Champions
   - Focus on retention of highest-value segment

5. **Conversion Optimization**
   - Focus on Email channel (2.8% conversion)
   - Optimize mobile experience (lower conversion than desktop)
   - Test new creative for Display ads (lowest ROAS)

---

## 🎯 Future Enhancements

### Planned Improvements

- [ ] Real-time dashboard with live data updates
- [ ] Deep learning models for improved churn prediction
- [ ] Product recommendation engine
- [ ] Customer lifetime value forecasting
- [ ] Geographic analysis by country/region
- [ ] Sentiment analysis of customer reviews
- [ ] Automated anomaly detection
- [ ] Marketing mix modeling (MMM)

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle - E-Commerce Marketing & Analytics Dataset
- **Inspiration**: Real-world marketing analytics challenges
- **Tools**: Python, Tableau, Scikit-learn communities

---

## ⭐ Show Your Support

If you found this project helpful, please give it a star! ⭐

---
