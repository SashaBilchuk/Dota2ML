# Dota2ML
With the rise of e-sports in recent years,
vast amount of data is being generated every day and is waiting to be analysed.
This creates significant value and opportunity in predicting in-game events using Machine Learning.

In this project Uri Moser and I used Clarity ( https://github.com/skadistats/clarity ) parser to transform data from
matches from raw binary code from over 2000 pro-scene matches into tabular data consisting of ober 2700 features of
the game state each 10 seconds.
We tried to predict wether there will be a team-fight in nearest time given sequence of last few minutes of the game.
We used some regular ML algorithms such as RandomForest, Gradient Boosting, Decision Tree, and AdaBoost Regressors and compared the to
Feedforward Neural Networks using Pytorch library.
The full review of the project can be found at the Dota_data_project.pdf

In order for it to work, one must also download processor.jar avliable from this link: 
https://drive.google.com/file/d/1BSQmK9N1-BISB_tAuZaiSbEztPxDwFw0/view?usp=sharing 
You'll need to put in the same direction with the python files
Notice it is a changed version of the Clarity parser modifyed for our needs. 
You can calibrate it for different time gaps and select different features.

Enjoy
