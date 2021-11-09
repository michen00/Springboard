# [Step 10: Pick Your Deployment Method](https://www.springboard.com/workshops/ai-machine-learning-career-track/learn#/curriculum/18201)

A short list of your next steps regarding deployment and engineering

I want to build a web app with a URL I can easily share with potential employers. In terms of basic requirements, it needs to 1) accept microphone input and 2) predict and display an inference with > 70% success. Sensible goals could be higher scores and a nice interface. Stretch goals could include continuous deployment, data gathering features, or a feature augmentation pipeline.

I'm thinking of building the Python app with [Streamlit](https://streamlit.io/). For the microphone component, I'm looking into building a Streamlit component with a JavaScript front end using the [WebRTC](https://webrtc.github.io/samples/src/content/peerconnection/webaudio-input/) or [MediaStream Recording](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API) APIs. For deployment and hosting, I'm considering [Heroku](https://www.heroku.com/about) and/or [Streamlit Cloud](https://streamlit.io/cloud).

The user will tap/click a recording button and speak into the edge device's microphone. The inference pipeline will need to pass the recorded audio to the Python model in Streamlit. Intermediate preprocessing steps will include trimming silences and audio standardization (bit rate, .wav, 16-bit PCM, etc.) before the FRILL embedding is extracted. Then the FRILL vector would be MinMax scaled and possibly undergo dimensionality reduction using prefit components. The model will then make a softmax prediction, which will be rendered as a simple bar plot. To avoid potential liability or human subjects concerns, no user data will be stored at this time.
