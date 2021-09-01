# Fin_summ
financial temporal summaries

To compile the project download the data from here https://drive.google.com/drive/folders/18kirMeMKdTmwtJ0vVB7X5KUJxcaZkzJ5?usp=sharing

And put in the same folder of the code. 

Run code.py first

Then a Folder named Results should appear, in this folder you find the subfolder df with the discretized financial series.

Run d2v.py to vectorize the financial series, you should then find in results the folders with the embedding per time span. 

Run then Rank_new.py to generate the ranking from the embedding considering the reference profile, the ranking is saved in a file named rank_new1.p and rank_new.json
these files are the same, you actually can find already this files in the git as i uploaded all the file that I could. 

after this you can run the summary you prefer, 
the summaryN.py file generates either a xlsx or a csv containing the results, dipending by its dimension, note that the summary4.py takes a long to run, it is suggested to 
generate just a sample of the results, this can be done by restricting the number of stocks at line 37 by doing: for stock1 in tqdm(list(stocks)[:RESTRICTEDNUMBER]): 
where restricted number is an integer spanning from 0 to 500. 
