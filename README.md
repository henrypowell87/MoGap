# MoGap
A deep learning application for gap filling motion capture data.

MoGap looks to use denoising autoencoders to gap fill motion capture data.
We use the CMU mo-cap dataset (here: http://mocap.cs.cmu.edu/) and a number
of different auto encoder architectures to fill in simulated missing data.

This is still a work in progress but current best results come from our CNN LSTM model 
which beats state-of-art models run through the same training process. More work needs
to be done to validate these results, however.
