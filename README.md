# About
The aim of this project is to compare 2 Recurrent Neural Network (RNN) architectures, namely the Long Short-term Memory
(LSTM) and Gated Recurrent Unit (GRU) in sentiment classification. Both architectures were designed to tackle the issues of the
traditional RNN namely the exploding and vanishing gradient problem which plagues longer sequence dependencies. They both
utilize gated units, but with different internal approaches to tackling these issues, hence why they form an interesting basis for
comparison.

The particular sentiment task looked at in this report is around Twitter sentiment classification, based on a [publicly available dataset](https://www.kaggle.com/kazanova/sentiment140).

![twitter sentiment analysis](https://user-images.githubusercontent.com/71750671/183037780-01f33d97-9c9e-4e00-b181-518a04ba95e0.png)

This project walks through the data pre-processing activities in preparing these tweets appropriately for LSTM and GRU classification,
and presents the steps taken in training and evaluating the LSTM and GRU models. The short summary of the results are presented below; for more details please refer to the report.

# Results and Conclusion Summary
From the metrics obtained in the the testing and evaluation phase, the following observations can be made:

1. The LSTM produces slightly more accurate results than the GRU
2. The LSTM converges sooner than the GRU
3. The GRU faster to train and evaluate than the LSTM

A case could be made for either model as the LSTM achieves slightly better results, but is more complex, whereas GRU is faster
and more memory efficient to train but is not as accurate as the LSTM. However, in choosing a single RNN architecture to perform,
twitter, sentiment analysis the GRU would have to be the more favourable choice. The GRU boasts a significantly faster training
time whilst achieving metrics essentially on par with its more complex LSTM competitor. This is mostly due to this particular
sentiment classification task whereby we find tweets taking upon a short sequence of words playing into the particular strengths of
the GRU. As this report indicates, the trade-off between minor accuracy lost in favour of a considerable time efficiency boost is
well worth it, thus crowning the GRU as the most suitable RNN architecture for Twitter sentiment analysis classifier.

## Dataset Acquisition
PLEASE NOTE: Due to size constraints, I have removed the "sentiment_dataset.csv" and "glove.6B.300d.txt" files.

Should you wish to run the program, these files will need to be obtained.
The "sentiment_dataset.csv" can be found at this [Google Drive link](https://drive.google.com/file/d/1YcHRhzekdw4urckdjJa-fWb5YC5ZrXjJ/view?usp=sharing)
or from [kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

The "glove.6B.300d.txt" file can be found at this [Stanford link](http://nlp.stanford.edu/data/glove.6B.zip)
Once unzipped, only the "glove.6B.300d.txt" file is required.
