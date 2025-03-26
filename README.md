# XGBoost Self-implementation

This is just a simple Binary Classification XGBoost model.

I was looking to familiarise myself with systems programming. Most of my experience prior to this was with regard to algorithms and ML. XGBoost implements quite a bit of these, so I thought coding it out myself would be a nice way to learn.


### SIMD & CUDA

Naturally I also thought it would be nice to try to implement some of these technologies. I didn't achieve the success I envisioned, because most of the parallelisable tasks had some sort of divergence, and ultimately the overhead in the setup usually outweighed the gains I got. 

Also I could only really test the CUDA stuff on Colab which made it quite a painful process.
