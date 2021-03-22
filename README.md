# Electricity Forecast
### Two type of ways to pass parameters.
```
python app.py [-h] [-p | -t] [-ip INPUTPREDICT] [-im INPUTMODEL]
              [-it INPUTTRAINING] [-o OUTPUT]
```
* Training model with the input csv file

-t : training
-it : the csv file that are going to throw into the model to train
-o : output file for model

```python
#Example
python app.py -t -it "本年度.csv" -o "100output82.h5"
```
* Predicting and get submission csv file from the model you choose
-p : predict
-ip : the csv file that are going to throw into the model and predict
-im : the model 
-o : output file (submission.csv)

```python	
#Example
python app.py -p -ip "本年度.csv" -im "100output82.h5" -o "submission.csv"
```