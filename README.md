# Background and Datasets
This project was an exploration of Neural Networks. It was done with Cameron Bronson, but I did all of the coding and led the project. He helped me analyze the results, create the report, create a PowerPoint that went along with it, and run my code on his computers.
The link to the smoking/drinking dataset: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset. The link to the original (uncleaned) powerlifting dataset: https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/.
These datasets were too large to directly upload to GitHub. The three datasets from Kaggle had 300,000-1,400,000 samples with many features; the generated datasets are considerably smaller. In this project, we chose to delete all samples with missing information. The project also features no pre-processing methods such as PCA; it aims to use complete, raw, unfiltered data.

# The Project and Files
The goal of this project was to explore and learn how to implement Neural Networks. We compared the accuracy of NN models against traditional methods. I used PyTorch for Neural Networks and sklearn for the other methods.
FinalReport.pdf has a detailed description of the project, how everything was implemented, downfalls and limitations, results, and discussions along with a list of references. The relevant files are SmokingDrinkingCleanUp, GeneratedDataSet, DiabetesCleanUp, and OpenPowerliftingClassifications. The Powerlifting Dataset had the extra file OpenPowerliftingCleanUp to save time from having to run the cleanup process multiple times.

Note these files will take hours to run as they currently are (unless you have lots of RAM, a good CPU, and many GPUs or have access to cloud computing). I recommend editing the training/testing split size down to a few hundred samples if you want to run any of the files.

# Future Hopes from this Project (Things I hope to Fix in Furutre Projects)
This semester (Spring 2024), I doing another ML project with a group of 5 people. I hope to explore new ML/DL algorithms and compare them to past algorithms. I also want to address some downfalls of this project that I would like to fix for future projects.

### Data Pre-Processing
I would like to implement more unsupervised learning techniques before going to supervised learning. This way we have a better feel for the data and would help to pick model hyperparameters. This semester, my ML class has a large portion dedicated to unsupervised learning, so I will hopefully learn many techniques for my next project.

### Data Visualization
To go along with pre-processing, I would also like to add more data visualization. Not only visualization of the data itself but visualization of the results. In the report for this project, we listed the results in a few paragraphs with lots of words. I think it would be much more helpful and neater to create graphs to easily see the results given by each method.
