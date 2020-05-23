## About me

Hello there! My name is Sebastian Scherer, and I'm working as a Data Scientist, but in my free time I enjoy doing small (or not so small) coding projects like this one.

If you like, please [stop by my Github](https://github.com/SebastianScherer88/) and check out some of my other content - there's even a playable Star Wars game in there!

For feedback or suggestions on how to improve this app, please reach out: scherersebastian@yahoo.de.

## About this app

### Hosting

The GUI (Graphic User Interface) you are seeing in front of you was written with R package [shiny](https://shiny.rstudio.com/). At the time of writing, it is hosted on one of the smaller AWS machines via the community editions of [Rstudio server](https://rstudio.com/products/rstudio/download-server/) and [R shiny server](https://rstudio.com/products/shiny/shiny-server/).

### Backend

The backend, i.e. the logic actually solving your input sudoku, is wrapped in a RestAPI written in python and the [fastAPI package](https://fastapi.tiangolo.com/). It has fantastic documentation and great integration with other related packages. 

This RestAPI is also hosted on the same AWS machine, but only locally.

### Resources and credits

While I wrote most of this application and its dependencies myself, I'm also rehashing some other people's code and resources, so I want to mention my sources and give credit where it's due.

#### R shiny and AWS

A great post to get started with R shiny on AWS is [this one by Jonas Schroeder](https://towardsdatascience.com/how-to-run-rstudio-on-aws-in-under-3-minutes-for-free-65f8d0b6ccda). It's referencing [Louis Aslett's amazing AWS machine image library](https://www.louisaslett.com/RStudio_AMI/) that he has made publically available. It makes it quick to set up an AWS machine that lets you host publically accessible R shiny apps like this one.

#### Mixed integer programming

The solving of a sudoku given appropriate initial values is a mixed integer programming problem. A there's a [great post by Allison Morgan](https://towardsdatascience.com/using-integer-linear-programming-to-solve-sudoku-puzzles-15e9d2a70baa) on how to implement a solution in python using the [pulp library](https://coin-or.github.io/pulp/). That code, only slightly amended, is used to solve the sudoku problem given by the user.
