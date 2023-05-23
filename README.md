# SF_Parking
This capstone project was done in collaboration with Jeffrey Kuo, Bryan Wang, and Tessa Weiss at the University of California, Berkeley for Stat 222 under the guidance of Professor Thomas Bengtsson.

## Central Goal and Project Introduction
The initial goal of our project was rigorously answer the following two questions about San Francisco (SF) parking citations. The first being: what is the probability of receiving a parking ticket given a time and place in San Francisco given that the individual is committing an infraction? The second being: are these citations being given out fairly? Or to put it into other terms, would one be less likely to receive a parking ticket if they are parked in a wealthier neighborhood? This readme will go over how these questions were answered.

## Noteboook Table of Concents

* eda.ipynb: Combined file compiling interesting EDA from each group member for presentation.
* eda_[name].ipynb: EDA done by respective group member.
* final_probabilities.ipynb: Calculating numerator divided by denominator.
* initial_kernel.ipynb: Imports / Helper Functions / Global Variables. This was also used to calculate the numerator.
* initial_poisson.ipynb: Poisson regression model that was used as baseline model. Not incorporated in final analysis.
* meter_eda.ipynb: EDA for meter datasets.
* meter_route_eda.ipynb: More specific meter EDA.
* meters.ipynb: Estimating the denominator.
* path_pred_prototype_tim: Trying to predict enforcement route. Not incorporated in final analysis.
* preprocess.ipynb: data preprocessing for Citations and streetsweeping dataset.
* reformat_table.ipynb: formatting table for web app.

The analysis pipeline should follow the following order:

eda files -> ppreprocess.ipynb -> initial_kernel.ipynb -> meters.ipynb -> final_probabilities.ipynb -> reformat_table.ipynb