# boxing-punch-classification
Bachelor thesis on boxing punch classification with accelerometer data

# Abstract
This thesis project evaluates the performance of five Machine Learning and two Deep Learning models on a human movement recognition task. Specifically, these algorithms were compared for their ability to classify boxing punches, captured by accelerometers. The acceleration of 7606 punches in the x, y and z axes, performed by eight professional boxing practioners, were used in a constructed Machine Learning (ML) pipeline and Deep Learning (DL) pipeline to train five supervised
Machine Learning models and two supervised Deep Learning models. By extracting a total number of thirty features from the accelerometer data, we were able to train ML algorithms that can accurately classify boxing punches with a minimal error rate. Giving the sequential accelerometer data as input to the DL models yielded similar results. Additionally, we investigate whether standardizing the period length of accelerometer data, influences the performance of these models.


The SmartPunchDataset has been acquired from TheSmartPunchTeam (2019) through https://www.kaggle.com/. Part of the code has been adapted by the me from Wagner (2019). The reused/adapted code fragments are clearly indicated in the notebook.


# References
Wagner, T. (2019). standardizer.py. GitHub. Retrieved from https://github.com/smartpunch/timeseries_helpers/blob/master/standardizer.py
