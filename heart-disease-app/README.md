CARDIOCARE AI - PROFESSIONAL UPGRADE

FILES
- app.py: complete upgraded Streamlit application; preserves the original modules and model-loading approach.
- requirements.txt: Python dependencies.

IMPORTANT MODEL FILES
Copy these three existing files from your old project into this same folder:
- RF_KNN_heart.pkl
- scaler.pkl
- columns.pkl
Do not rename, retrain, overwrite or edit them.

WINDOWS SETUP
1. Install Python 3.10 or newer.
2. Extract this folder to a normal location, for example Desktop\CardioCare_AI_Professional.
3. Copy the three .pkl files into the folder beside app.py.
4. Open the folder in VS Code.
5. Open Terminal > New Terminal.
6. Create a virtual environment:
   py -m venv venv
7. Activate it:
   venv\Scripts\activate
8. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt
9. Run:
   streamlit run app.py
10. The browser should open the application.

OPTIONAL API KEYS
The existing AI Health Assistant can use OPENAI_API_KEY and optional TAVILY_API_KEY.
For local testing, set them as environment variables or Streamlit secrets. Never paste API keys into app.py or publish them to GitHub.

REPORT UPLOAD
- PDF text extraction uses pdfplumber.
- Image OCR uses Pillow + pytesseract and also requires installing the Tesseract OCR application on Windows.
- If a scanned PDF contains images rather than selectable text, OCR is not automatically applied to its pages in this first implementation; save the page as an image or use a searchable PDF.
- Check every extracted value against the original report before using it.

LOCATION FINDER
The location finder asks the user to type an area/city/PIN code. It uses public OpenStreetMap Nominatim and Overpass services. Results may be incomplete or outdated; phone numbers, availability, prices, home collection and bookings must be confirmed with the provider. The app does not dispatch a technician or make a booking.

DATA / PRIVACY WARNING
This is a development/college-project prototype. It stores report metadata and extracted text in a local SQLite file named cardiocare_clinical.db and prediction history in prediction_history.csv. These local files are not encrypted. Do not upload real patient-identifiable records or deploy for real clinical use until authentication, encryption, authorization, secure hosting, backups, consent, audit logging and applicable Indian privacy/medical-device compliance have been implemented and reviewed.

MEDICAL SAFETY
The ML output and report explanations are not diagnoses. The existing model's score is not necessarily calibrated to real-world risk. A normal-looking result does not rule out disease. Emergency symptoms require immediate medical care, not app-based screening.
