# Machine Learning AdBlocker
**By Klaas Schoenmaker and Marjolijn Stam**

This program uses over 100.000 links crawled from the top-500 most-visited websites. After being trained with AdBlock Plus's EasyList, our model is able to predict whether a link is an ad-related link or not with 95% accuracy.

### Try the model

You can try the model by running the prediction script that uses a saved version of the model (`data/trained_model.pkl`).
You pass the URL to block as a parameter to the script.

For example, run:

```sh
$ python predict.py "https://www.google.com/"
```