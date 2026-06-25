# SportSlot Pro

SportSlot Pro is a Streamlit-based sports venue booking web app. It lets users browse a premium sports venue page, view available sports and amenities, optionally log in or skip login, choose booking details, get an estimated slot price, book a spot, and complete a simulated payment flow.

## Features

- Premium glassmorphism UI with an obsidian and neon-lime theme
- Scrollable venue landing page with gallery, sports, amenities, about, and booking sections
- Optional login/signup flow with skip-login support
- Booking form with sport, date, start hour, duration, players, court type, equipment rental, and contact person
- Date picker that automatically derives the day of week and weekend status
- Machine learning price estimation trained from `bookings_dataset.csv`
- Book spot and payment confirmation workflow
- Horizontally scrollable sports carousel

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- HTML/CSS for custom UI styling

## Project Structure

```text
.
├── app.py                  # Main Streamlit application
├── analysis.ipynb          # Notebook used for EDA and model experimentation
├── bookings_dataset.csv    # Sports booking dataset
├── requirements.txt        # Python dependencies
└── assets/                 # UI images used in the app
```

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dataset

The app uses `bookings_dataset.csv`, which contains sports booking records with fields such as:

- Sport
- Booking date
- Day of week
- Weekend flag
- Start hour
- Duration
- Number of players
- Court type
- Equipment rental
- Booking price

## Model

The app trains a price estimator at runtime using the dataset. The model is used internally to estimate the booking price from the user-selected slot details.

## Notes

- This is a demo booking application. Payment and booking confirmation are simulated inside Streamlit.
- Keep `bookings_dataset.csv` in the project root so the app can load and train correctly.
- Keep the `assets/` folder with the app if deploying or sharing the project.

## Screens

- Venue landing page
- Sports and amenities sections
- Login or skip-login flow
- Slot booking form
- Price and checkout panel

