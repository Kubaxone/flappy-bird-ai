# Flappy Bird AI Evolution

A local Flappy Bird game where a population of neural networks learns through NEAT neuroevolution. The game uses the prepared art in `sprites/`, displays opaque and translucent trajectory replays, and saves checkpoints and the all-time best genome under `training_data/`.

## Run

```powershell
pip install -r requirements.txt
python main.py
```

The latest NEAT checkpoint resumes automatically. **R** starts a fresh population while preserving the all-time best record. **SPACE** pauses, **UP/DOWN** changes simulation speed, and **ESC** exits. A checkpoint is saved at the end of each generation and on exit.

## Learning

Each genome receives five normalized observations: bird Y, vertical velocity, distance to the next pipe, gap top, and gap bottom. Its output above 0.5 triggers a flap. Fitness grows for surviving and staying near the upcoming gap, gets a larger reward for passing pipes, and loses fitness on collision. All 60 genomes are evaluated together in the same game world; translucent birds are trajectory replays only. The current generation's best fitness is plotted live.

The **current bird** is opaque, the **previous run** is cyan at 40% opacity, and the **all-time best** is gold at 70% opacity with a crown. The best genome, best score, fitness history, and recorded best trajectory persist across launches. The window uses a 9:16 vertical layout and shows only the generation, highest score, and current score. Stored trajectory points are bounded to reduce memory use. Supplied sounds play for the visible bird's flap, pipe pass, pipe hit, and ground crash; audio disables itself automatically when no output device is available.

## TikTok LIVE

At startup, the app attempts to listen to `@kubaxone`. Five distinct accounts posting exactly `!changebg` trigger a one-second fade between the supplied day and night backgrounds. Chat is optional; training works without a live connection. TikTok has no public official live-chat API, so this uses the third-party TikTokLive package; comments arrive only while the streamer is live and the client can connect.


README WRITTEN BY AI!

Before you delete training_data
I recommend making a backup
if you wanna train it yourself delete master_data 
if you want it to use the training data it was fed delete training_data and rename master_data to training_data then it should be a master at the game
before anyone gets mad at me for not saying i used ai in this project
AI WAS USED
mainly to fix bugs/implement sprites