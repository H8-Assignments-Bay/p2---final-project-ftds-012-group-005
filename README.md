<!-- [![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8070902&assignment_repo_type=AssignmentRepo) -->

<div align="center">

# JoFi (JourneyFinder)

[Introduction](#introduction) • [Installation](#installation) • [Screenshots](#demonstration-on-telegram) • [Contributors](#contributors)

![GitHub repo size](https://img.shields.io/github/repo-size/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub last commit](https://img.shields.io/github/last-commit/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub contributors](https://img.shields.io/github/contributors/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub top language](https://img.shields.io/github/languages/top/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)

This ***Project*** is a ***Final Project*** for Hacktiv8 Full Time Data Science Program.

![logo 16x9](https://raw.githubusercontent.com/H8-Assignments-Bay/p2---final-project-ftds-012-group-005/main/logo/banner.png)
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#JoFi(JourneyFinder)">JoFi (JourneyFinder)</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- Introduction -->
# Introduction

**Traveling** is an activity that is much favored by the public, usually people who like traveling are referred to as travelers. `Information about traveling` is very important for travelers, especially when visiting new places that have never been visited, therefore the information must be accurate and complete. Every traveler has different characteristics depending on their personality and preferences. There are mountain ⛰️ hikers, beach 🌊 hunters and etc. Therefore, through ***JoFi*** we build a ***hidden gem recommender system*** based on `user preferences`. *With JoFi, find your journey is never this easy*.

<!-- USAGE EXAMPLES -->
# Usage

With this app, you can ***just upload an image***, or ***describe the place where you want to go***. When you've decided the place, then there is a feature `traffic-sign classifier` on the app that can classify the `unfamilar traffic-sign` specially when you are a foreigners.

_For more examples, please refer to the [Documentation](https://journeyfinder.herokuapp.com/)_

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
# Roadmap

- [x] Add Dataset (scrapped)
- [x] Build Models
- [x] Build Web Application
- [ ] Add more complete data (the size of origin data is too small ->  affecting the model performance)
- [ ] Add Indonesian Traffic Sign Data (we only use German Traffic Sign Data on this project)
- [ ] Add more features (user preferences as traveller or guider)
- [ ] Add More Recommendations / Hidden Gem
- [ ] Multi-language Support

See the [open issues](https://raw.githubusercontent.com/H8-Assignments-Bay/p2---final-project-ftds-012-group-005/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
# Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this project better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b any/improvements`)
3. Commit your Changes (`git commit -m "Add some improvements"`)
4. Push to the Branch (`git push origin any/improvements`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTORS -->
# Contributors

This project is created as a collaboration project between:


- **Marwan Musa** 
[<img align="left" alt="wahyudi linkedin" width="22px" src="https://user-images.githubusercontent.com/103250002/182405954-d5ae3cda-d74c-43a5-b995-8220005d151f.gif" />][mm] 
[<img align="left" alt="wahyudi github" width="22px" src="https://user-images.githubusercontent.com/103250002/182372179-0954a140-ddbb-489d-83a7-51c715d7ae6d.svg" />][ml]



- **Rio Armiga** 
[<img align="left" alt="enggar linkedin" width="22px" src="https://user-images.githubusercontent.com/103250002/182405954-d5ae3cda-d74c-43a5-b995-8220005d151f.gif" />][rm] 
[<img align="left" alt="enggar github" width="22px" src="https://user-images.githubusercontent.com/103250002/182372179-0954a140-ddbb-489d-83a7-51c715d7ae6d.svg" />][rl]


[mm]:https://www.linkedin.com/in/marwanmusa/
[ml]:https://github.com/marwanmusa
[rm]:https://www.linkedin.com/in/rio-armiga/
[rl]:https://github.com/rioarmiga

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- REFERENCES -->
## References
### *Image Classifier*
>   https://www.kode.id/ (***by Hacktiv8***)
    https://colab.research.google.com/github/ardhiraka/FSDS_Guidelines/blob/master/p2/w2/d1pm.ipynb#scrollTo=hhIFMAfa3Igq (***Hacktiv8 material course***) <br>
    https://www.kaggle.com/code/avikumart/computervision-intel-image-classification-project/notebook <br>
    https://www.kaggle.com/code/janvichokshi/transfer-learning-cnn-resnet-vgg16-iceptionv3 <br>
    https://www.kaggle.com/code/mjain12/intel-image-classification-cnn-vgg16 <br>
    https://www.myaccountingcourse.com/accounting-dictionary/f1-score <br>
    https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc <br>

### *Traffic-sign Classifier*
>   https://benchmark.ini.rub.de/ <br>
    https://github.com/yassiracharki/DeepLearningProjrects <br>
    https://www.kaggle.com/code/saeidghomi/gtsrb-by-cnn <br>
    https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn <br>
    https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean#:~:text=In%20a%20CNN%20context%2C%20people,kernel%22%20is%20the%20filter%20itself. <br>

### *Text Classifier*
>   https://www.kaggle.com/code/aayush895/text-classification-using-keras <br>
    https://www.kaggle.com/code/aashita/word-clouds-of-various-shapes/notebook<br>
    https://www.kaggle.com/code/junedism/spaceship-titanic-exploratory-data-analysis
    https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/<br>
    https://machinelearningmastery.com/clean-text-machine-learning-python/<br>
    https://keras.io/api/preprocessing/text/#one_hot<br>
    https://www.analyticsvidhya.com/blog/2021/11/an-introduction-to-stemming-in-natural-language-processing/<br>
    https://www.datacamp.com/tutorial/wordcloud-python<br>
    https://coderpad.io/regular-expression-cheat-sheet/<br>
    https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39<br>
    https://www.goeduhub.com/10643/practical-approach-word-embedding-simple-embedding-example<br>
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences<br>
    https://colab.research.google.com/drive/1quIzzM4444f41LvSGMIDzNhZbBggUFZo#scrollTo=UbOU7Zh_RmXK (***hacktiv8 material course***)<br>
    https://www.tensorflow.org/tfx/tutorials/transform/census<br>
    https://www.kaggle.com/code/sardiirfansyah/tensorflow-input-pipeline-prefetch-tf-data<br>
    https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle<br>
    https://medium.com/@ashraf.dasa/shuffle-the-batched-or-batch-the-shuffled-this-is-the-question-34bbc61a341f<br>
    https://stackoverflow.com/questions/56227671/how-can-i-one-hot-encode-a-list-of-strings-with-keras<br>
    https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e<br>
    https://keras.io/api/layers/core_layers/embedding/<br>
    https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm<br>
    https://www.kaggle.com/code/fanyuanlai/textcnn<br>
    https://www.kaggle.com/code/tanvikurade/fake-job-postings-using-bidirectional-lstm/notebook<br>
    https://medium.com/deep-learning-with-keras/lstm-understanding-the-number-of-parameters-c4e087575756<br>
    https://medium.com/geekculture/10-hyperparameters-to-keep-an-eye-on-for-your-lstm-model-and-other-tips-f0ff5b63fcd4<br>
    https://towardsdatascience.com/lstm-framework-for-univariate-time-series-prediction-d9e7252699e<br>
    https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359<br>
    https://medium.com/ai-ml-at-symantec/should-we-abandon-lstm-for-cnn-83accaeb93d6<br>
    https://analyticsindiamag.com/guide-to-text-classification-using-textcnn/<br>
    https://keras.io/api/layers/pooling_layers/max_pooling1d/<br>

### *Chatbot*
    Full tutorial on NgodingPython YouTube Channel https://www.youtube.com/watch?v=sotu6YqPoY0 <br>
    https://github.com/H8-Assignments-Bay/p2---final-project-group-004/blob/main/gitcoff_bot.py <br>
    https://github.com/gcatanese/TelegramBotDemo <br>
    https://towardsdatascience.com/bring-your-telegram-chatbot-to-the-next-level-c771ec7d31e4 <br>


<p align="right">(<a href="#top">back to top</a>)</p>