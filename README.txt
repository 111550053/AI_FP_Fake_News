to train a model
create a folder "data" and put the dataset which is going to be used to train in it.
python3 main.py
or
run project.ipynb

to test a model
create a folder "data" and put the dataset which is going to be used to test in it.
create a folder "model" if you've had a model "project.pt" and put the model into the folder.
python3 testwithmodel.py

to use LIME to explain the model
put the folders "model" and "data" to google drive
run blackbox.ipynb
find the last blcok of blackbox("...") and put the article between ""
note that the " in the article should be modified to \"
then run the block
