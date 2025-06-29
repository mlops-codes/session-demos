Create Py Env
    python -m venv venv 

Activate Env
Windows
    venv/Scripts/activate

Mac/Linux
    source venv/bin/activate

Create Model 
     python .\model-demo\model.py

Run Streamlit App (make sure .pkl is present)
    cd model-demo
    streamlit run .\loan_app.py