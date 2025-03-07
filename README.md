# Company Classification Tool

A Streamlit application that analyzes companies to determine if they have a market-based or domain-based orientation, identifies competitors, and provides strategic recommendations.

## Features

- **Company Analysis**: Evaluates if a company is market-based or domain-based on a scale of 0-10
- **Competitor Identification**: Identifies top competitors based on company description
- **Comparative Analysis**: Compares your company with competitors on the market/domain spectrum
- **Strategic Recommendations**: Provides tailored recommendations based on analysis results
- **Visual Reports**: Includes charts and visual indicators for easy interpretation
- **Export Functionality**: Download analysis results as JSON

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Create a file with your Groq API key
   - Create a file with your Serper API key (optional)

## Usage

1. Run the application:

```bash
streamlit run prototype.py
```

2. Enter your company information:
   - Company name
   - Company description
   - Mission statement
   - Vision statement

3. Click "Analyze Company" to start the analysis

4. Review the results in the following sections:
   - Company Orientation Analysis
   - Competitor Analysis
   - Comparative Analysis
   - Strategic Recommendations

## Definitions

- **Market-based View**: Focused on becoming a leader in a particular industry, product category, or market segment. Companies with this orientation typically focus on existing market needs, are reactive to market trends, and prioritize meeting current customer requirements.

- **Domain-based View**: Focused on addressing human-centric needs across different industries, product categories, or market segments. Companies with this orientation typically develop deep expertise in specific domains, create new markets through innovation, and are more proactive than reactive.

## Requirements

- Python 3.8+
- Groq API key
- Internet connection

## License

[MIT License](LICENSE) 