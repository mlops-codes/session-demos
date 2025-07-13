install packages - 
pip install -r .\requirements.txt

Train Model -
python .\demo-2\train_model.py   

Run API -
python .\demo-2\api.py   

CURL to test api
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{ "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 }