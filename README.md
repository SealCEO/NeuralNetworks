# Background and Datasets
This project was an exploration of Neural Networks. It was done with Cameron Bronson, but I did all of the coding and lead the project. He helped me analyze the results, create the report, create a powerpoint that went along with it, and ran my code on his computers.
The link to the smoking/drinking dataset: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset. The link to the original (uncleaned) powerlifting dataset: https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/.
These datasets were too large to directly upload to GitHub. The datasets had 300,000-1,400,000 samples with many features. In this project, we chose to delete all samples with missing information. The project also features no pre-processing methods
such as PCA; it aims to use complete, raw, unfiltered data.

# The Project and Files
The goal of this project was to learn how to explore/implement Neural Networks and compare their accuracy against traditional methods across many scenarios. I used PyTorch for the implementation of Neural Networks and sklearn for the other methods.
FinalReport.pdf has a detailed description of the project, how we specifically implemented everything, downfalls and limitations, results, and discussions along with a link of references to our research. The python files implementing the methods are SmokingDrinkingCleanUp, GeneratedDataSet, DiabetesCleanUp, and OpenPowerliftingClassifications. The Powerlifting Dataset had the extra file OpenPowerliftingCleanUp to save time from having to rerun the cleanup process each run.

Note these files will take hours to run as they currently are (unless you have lots of RAM, a good CPU, and many GPUs). Edit the training/testing split size down to a few hundred samples if you wish to run any of the files.
