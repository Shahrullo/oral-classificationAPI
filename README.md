# Oral Classification API
A RESTful API that classifies if images captured is oral or non-oral.


## How To Run Locally 
### Option 1: Using Python `virtualenv`
You can use the following commmand to create a new Python 3 virtual environment (any version of Python 3.5–3.8 should work, because we are using Tensorflow 2):
```
python -m venv environment_name
``` 
Afterwards you can install the dependencies via `pip`:
```
(env) python -m pip install -r requirements.txt
```
Finally you can configure the environment variables and run the Flask module:
```
(env) export FLASK_ENV=development; export FLASK_APP=application; flask run
```
### Option 2: Using Docker




