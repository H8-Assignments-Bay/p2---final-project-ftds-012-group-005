<!-- [![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8070902&assignment_repo_type=AssignmentRepo) -->

<div align="center">

# JoFI (Journey Finder)

[Introduction](#introduction) • [Installation](#installation) • [Screenshots](#demonstration-on-telegram) • [Contributors](#contributors)

![GitHub repo size](https://img.shields.io/github/repo-size/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub last commit](https://img.shields.io/github/last-commit/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub contributors](https://img.shields.io/github/contributors/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)
![GitHub top language](https://img.shields.io/github/languages/top/H8-Assignments-Bay/p2---final-project-ftds-012-group-005)

This project is created as a collaboration by a group of students in Hacktiv8 Full Time Data Science Program.

![logo 16x9](https://raw.githubusercontent.com/H8-Assignments-Bay/p2---final-project-ftds-012-group-005/main/logo/banner.png)
</div>

---

### Table of Contents
- [JoFI (Journey Finder)](#fintbot---fintech-chatbot)
    - [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Demonstration on webapp](#demonstration-on-web)
- [Demonstration on telegram](#demonstration-on-telegram)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Running on Your Local Machine](#running-on-your-local-machine)
- [Features](#features)
- [Using your own data](#using-your-own-data)
- [Roadmap](#roadmap)
- [Rasa Open Source](#rasa-open-source)
- [Contributors](#contributors)

# Introduction

Traveling is an activity that is much favored by the public, usually people who like traveling are referred to as travelers. Information about traveling is very important for travelers, especially when visiting new places that have never been visited, therefore the information must be accurate and complete. Every traveler has different characteristics depending on their personality and preferences. Starting from mountain hikers (travelers who like to hike the mountain) to beach hunters (travelers who like to find a beautiful beach). Therefore, through JoFi we build a traveler attraction recommender system based on user preferences. With JoFi, Find your journey is never this easy.

<div align="center">

### Demonstration on telegram

![Fintbot-Demo2](https://user-images.githubusercontent.com/69398229/177583069-f7bbfd85-ab93-438e-9e04-1d972e8e80d6.gif)
</div>

# Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [GIt](https://git-scm.com/downloads)
# Installation

To run this chatbot on your own machine, first clone the repository and navigate to it:
```
git clone https://github.com/H8-Assignments-Bay/p2---final-project-group-003.git fintbot
cd fintbot
```
Installed files could reach ~3GB on local storage because of the hefty environment needed

## Environment Setup

Next, run the following command to create a `conda` virtual environment that contains all the libraries needed to run the chatbot:\
*For windows user, you might need to use `Anaconda Prompt`

```
conda env create -n fintbot -f ./environment.yml
```

# Running on Your Local Machine

Activate conda environment, using this command:
```
conda activate fintbot
```

Run the following command to run Fintbot on your terminal:
```
rasa run actions &>/dev/null &
rasa shell
```

Or you can also set it up to run on your favorite messaging application, like we did on Telegram.

# Features

These are the things you can do with Fintbot:
1. Initialization (Hi, Hello, etc)
2. Account Balance
3. Withdrawal with account balance reduction
4. Investment with personal account balance reduction and investment account balance update
5. Ask Fintbot for help to remind you of the features

# Using your own data

You can create a new model based on Fintbot using your own data. Simply edit the dataset in the `data` directory. Refer to [Rasa Documentation](https://rasa.com/docs/rasa/) for more details. Afterwards, you can create a new model by running the following code:

```
rasa train
```

# Roadmap

This chatbot is still in development, so we are working on adding more features and improving the performance of the chatbot. Such as:
- Adding a portfolio feature so users can track which projects they have invested in
- Using RDBMS to store the database, instead of using Rasa's slot system

# Rasa Open Source 

Huge Thanks to [Rasa Open Source](https://github.com/RasaHQ/rasa)

We used rasa open source to create our chatbot. Rasa is an open source machine learning framework for creating AI assistance and also chatbots. Rasa assists you construct logical assistance fit for having layered conversation with heaps of back-and-forth. For a human to have a significant trade with a context oriented assistance. 

# Contributors

This project is created as a collaboration project between:
1. [Annesa Fadhila Damayanti](https://github.com/Nurfaldi)
2. [M. Nurfaldi Rosal](https://github.com/nesafadhila)
3. [Nikki Satmaka](https://github.com/NikkiSatmaka)
