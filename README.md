

# Comprehensive Space Threat Prediction System 

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-green?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.0-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.13.0-purple?logo=plotly)
![NASA NEO API](https://img.shields.io/badge/NASA%20NEO%20API-v1.0-lightgrey?logo=nasa)

The **Comprehensive Space Threat Assessment and Prediction System (CSTAPS)** is a web-based application designed to analyze and predict potential threats from Near-Earth Objects (NEOs). It integrates NASA's NEO API with advanced machine learning models and physics-based simulations to provide real-time data analysis, interactive visualizations, and impact predictions.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [Technologies Used](#technologies-used)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)
9. [Contact](#contact)

---

## Features
- **📊 Real-Time Data Fetching**: Fetches real-time data from NASA's NEO API.
- **🌌 Interactive Visualizations**: Provides 3D trajectory visualizations and impact heatmaps using Plotly.
- **🤖 Machine Learning Models**: Uses Scikit-learn and XGBoost for threat prediction and impact probability estimation.
- **💥 Impact Simulation**: Simulates the effects of NEO collisions, including crater formation and atmospheric effects.
- **🖥️ User-Friendly Interface**: Built with Streamlit for a clean and intuitive user experience.

---

## Installation
To run the CSTAPS project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Space-Threat-Assessment-System.git
   cd Space-Threat-Assessment-System
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`.

---

## Usage
1. **📡 Real-Time Data Analysis**:
   - The application fetches real-time data from NASA's NEO API and displays it in an interactive dashboard.

2. **📈 Threat Prediction**:
   - Use the machine learning models to predict the probability of NEO impacts.

3. **💥 Impact Simulation**:
   - Simulate the effects of NEO collisions, including crater formation and atmospheric effects.

4. **📤 Data Export**:
   - Export data and visualizations in CSV, JSON, or Excel formats.

---

## File Structure
```
project/
├── app.py                 # Main application file
├── data_processing.py     # Data fetching and preprocessing
├── model_training.py      # Machine learning models and training
├── visualization.py       # Interactive visualizations
├── export.py              # Data export functionality
├── config.toml            # Configuration file for Streamlit and API keys
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## Technologies Used
- **🐍 Python 3.11**: Primary programming language.
- **🚀 Streamlit**: For building the web interface.
- **🤖 Scikit-learn**: For machine learning models.
- **📊 XGBoost**: For regression-based impact energy prediction.
- **📈 Plotly**: For interactive 3D visualizations.
- **🛰️ NASA NEO API**: For real-time NEO data.

---

## Contributing
We welcome contributions to the CSTAPS project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
- **🛰️ NASA**: For providing the NEO API.
- **🚀 Streamlit**: For the web framework.
- **🤖 Scikit-learn and XGBoost**: For machine learning capabilities.
- **📈 Plotly**: For interactive visualizations.

---

## Contact
For questions or feedback, please contact:
--**Tharun JS ** : tharunjs012003@gmail.com
- **Akash M**: akash@example.com
- **K.S. Ahmed Abrarul Hag**: abrarul@example.com
- **M Abinesh**: abinesh@example.com
- **M Hari Vignesh**: hari@example.com
```

---

