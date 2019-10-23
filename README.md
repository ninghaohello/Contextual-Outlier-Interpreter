# Contextual-Outlier-Interpretation

This project provides an implementation for the [paper](https://www.ijcai.org/proceedings/2018/0341.pdf): <br>

> **Contextual Outlier Interpretation**<br>
Ninghao Liu, Donghwa Shin, Xia Hu<br>
IJCAI 2018 <br>


### Files in the folder
- `data/`
  - `wbc/`: a example dataset used in outlier detection
    - `X.csv`: each line represents one instance;
    - `y.csv`: labels indicating whether each instance is an outlier (1) or not (0);
- `src/`: implementations of the proposed outlier explanation method
  - `main.py`: runs the proposed method on the dataset
  - `outlier_interpreter.py`: implementation of the interpretation method
  - `prediction_strength.py`: estimates the number of clusters
  - `utils.py`: includes LOF and IsolationForest as outlier detectors
