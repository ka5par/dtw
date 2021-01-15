# Trading with
Pattern matching trading algorithms which compare the performance of different similarity measures such as DTW/TWED/LCSS/Corr. 
The distances calculated by the similarity measures are used as an input for KNN.

Methodology is based on the papers by Nagakawa, Imamura and Yoshida [^1][^2]. 

[^1]: https://doi.org/10.1002/ecj.12140

[^2]: https://doi.org/10.1007/978-3-319-93794-6_7

-----------------------

Basic instructions to run this script:

Create a new python env and install all the dependencies from requirements.txt.

Files:

`update_data.py` - Downloads Yahoo/Investing data. Output is a csv file with columns Date & Close. 

`prediction.py` - Input: data in the folders&format provided by `update_data.py`, user inputs months given for out of sample (OOS) predictions.
Output: Calculates buy&sell predictions for OOS.

`inference.py` - Input: data in the folders&format provided by `prediction.py`. Output: Total returns plots and performance tables (total returns, alpha, accuracy) per input.

Functions:

`stat_model.py` - Contains statistical models (KNN, K*NN)

`distance_model.py` -  Contains distance models (DTW, TWED, LCSS, Corr)

Research:

`param_test*` - parameter tests ran for TWED&LCSS 
