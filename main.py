"""NEAT-trained Flappy Bird with persistent training and replay ghosts."""
import json
import pickle
import random
import sys
import time
from collections import deque
from pathlib import Path

import neat
import pygame
from tiktok_chat import TikTokChatListener

ROOT = Path(__file__).resolve().parent
SPRITES = ROOT / "sprites"
DATA = ROOT / "training_data"
CHECKPOINTS = DATA / "checkpoints"
STATE = DATA / "evolution.json"
BEST = DATA / "best_genome.pkl"
W, H, GH, GROUND = 360, 560, 150, 510
BX, PW = 88, 64
WINDOW_SIZE = (450, 800)  # 9:16, suitable for a vertical live-stream layout
SCENE_SIZE = (W + 32, H + 32)
PATH_SAMPLE_EVERY = 4
MAX_PATH_POINTS = 1200


class Bird:
    def __init__(self):
        self.y, self.v, self.alive = 240., 0., True
        self.visual_y = self.y
        self.score, self.fit, self.path = 0, 0., []
        self.passed_pipes = set()
        self.rewarded_100 = False

    def update(self, flap, frame):
        if flap:
            self.v = -7
        self.v = min(self.v + .42, 9.5)
        self.y += self.v
        if frame % PATH_SAMPLE_EVERY == 0 and len(self.path) < MAX_PATH_POINTS:
            self.path.append((BX, self.y, self.v))


class Pipe:
    next_id = 0

    def __init__(self, x):
        self.id = Pipe.next_id
        Pipe.next_id += 1
        self.x, self.mid, self.passed = float(x), random.randint(155, 360), False
        self.visual_x = self.x


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird AI Evolution")
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        self.screen = pygame.Surface(SCENE_SIZE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 27, bold=True)
        self.assets = self.load_assets()
        self.sounds = self.load_sounds()
        self.sound_last_played = {}
        self.speed, self.paused, self.running = 1, False, True
        self.previous_path, self.best_path, self.history = [], [], []
        self.best_genome = None
        self.best_fit, self.best_score, self.gen_fit, self.done_gens = 0., 0, 0., 0
        self.is_night, self.fade_from, self.fade_at = False, False, None
        self.messages, self.chatters, self.chat_status = deque(), set(), "connecting..."
        self.load_data()
        self.chat = TikTokChatListener("kubaxone", self.messages, self.status)
        self.chat.start()

    def load_assets(self):
        def load(name):
            return pygame.image.load(str(SPRITES / name)).convert_alpha()
        bgday = pygame.image.load(str(SPRITES / "background-day.png")).convert()
        bgnight = pygame.image.load(str(SPRITES / "background-night.png")).convert()
        bird = [load(f"yellowbird-{n}flap.png") for n in ("up", "mid", "down")]
        return {"day": pygame.transform.scale(bgday, (W, H)),
                "night": pygame.transform.scale(bgnight, (W, H)),
                "birds": [pygame.transform.scale(b, (34, 24)) for b in bird],
                "pipe": pygame.transform.scale(load("pipe-green.png"), (PW, 320))}

    def load_sounds(self):
        """Load supplied sound effects when the machine has an audio device."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            return {name: pygame.mixer.Sound(str(ROOT / "audio" / f"{name}.wav"))
                    for name in ("wing", "point", "hit", "die")}
        except (pygame.error, OSError):
            # Keep training usable on systems without an audio device.
            return {}

    def play_sound(self, name, cooldown=0.0):
        sound = self.sounds.get(name)
        now = time.monotonic()
        if sound is None or now - self.sound_last_played.get(name, 0.0) < cooldown:
            return
        sound.play()
        self.sound_last_played[name] = now

    def status(self, text):
        self.chat_status = text

    def load_data(self):
        DATA.mkdir(exist_ok=True)
        if STATE.exists():
            try:
                d = json.loads(STATE.read_text(encoding="utf-8"))
                self.best_fit = float(d.get("best_fitness", 0))
                self.best_score = int(d.get("best_score", 0))
                self.history = list(map(float, d.get("history", [])))
                self.done_gens = int(d.get("completed_generations", len(self.history)))
                self.best_path = [tuple(v) for v in d.get("best_path", [])]
                self.previous_path = [tuple(v) for v in d.get("previous_path", [])]
            except (OSError, ValueError, TypeError):
                pass
        if BEST.exists():
            try:
                with BEST.open("rb") as f: self.best_genome = pickle.load(f)
            except (OSError, pickle.PickleError, EOFError):
                pass

    def save_data(self):
        DATA.mkdir(exist_ok=True)
        temp = STATE.with_suffix(".tmp")
        temp.write_text(json.dumps({"best_fitness": self.best_fit, "best_score": self.best_score,
                                    "history": self.history, "completed_generations": self.done_gens,
                                    "best_path": self.best_path[-MAX_PATH_POINTS:],
                                    "previous_path": self.previous_path[-MAX_PATH_POINTS:]}), encoding="utf-8")
        temp.replace(STATE)

    def config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, str(ROOT / "config-feedforward.txt"))

    def new_population(self):
        p = neat.Population(self.config())
        p.add_reporter(neat.StdOutReporter(False))
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(neat.Checkpointer(1, filename_prefix=str(CHECKPOINTS / "neat-checkpoint-")))
        return p

    def population(self):
        conf = self.config()
        files = sorted(CHECKPOINTS.glob("neat-checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1])) if CHECKPOINTS.exists() else []
        if not files:
            return self.new_population()
        p = neat.Checkpointer.restore_checkpoint(str(files[-1]))
        p.config = conf
        p.add_reporter(neat.StdOutReporter(False))
        p.add_reporter(neat.StatisticsReporter())
        p.add_reporter(neat.Checkpointer(1, filename_prefix=str(CHECKPOINTS / "neat-checkpoint-")))
        return p

    def checkpoint(self):
        if not hasattr(self, "pop"):
            return
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "checkpoint_reporter"):
            self.checkpoint_reporter = neat.Checkpointer(1, filename_prefix=str(CHECKPOINTS / "neat-checkpoint-"))
        self.checkpoint_reporter.save_checkpoint(self.pop.config, self.pop.population,
                                                 self.pop.species, self.pop.generation)
        self.save_data()

    def events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: self.running = False
                elif e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key == pygame.K_UP: self.speed = min(30, self.speed + 1)
                elif e.key == pygame.K_DOWN: self.speed = max(1, self.speed - 1)
                elif e.key == pygame.K_r: self.restart = True

    def evaluate(self, genomes, config):
        agents = []
        for gid, genome in genomes:
            genome.fitness = 0.
            agents.append((genome, Bird(), neat.nn.FeedForwardNetwork.create(genome, config)))
        self.gen = self.pop.generation
        self.gen_fit, self.frame, self.restart = 0., 0, False
        self.pipes = [Pipe(W + 40)]
        while self.running and any(b.alive for _, b, _ in agents):
            self.events()
            self.chat_events()
            if not self.running or self.restart:
                self.checkpoint()
                return
            if self.paused:
                self.draw(agents)
                self.clock.tick(60)
                continue
            for _ in range(self.speed):
                self.step(agents)
                if not any(b.alive for _, b, _ in agents): break
            self.draw(agents)
            self.clock.tick(60)
        if agents:
            self.previous_path = max(agents, key=lambda a: a[1].fit)[1].path[:]
        self.history.append(self.gen_fit)
        self.history = self.history[-5000:]
        self.done_gens += 1
        if self.best_genome is not None:
            with BEST.open("wb") as f: pickle.dump(self.best_genome, f)
        self.save_data()
        self.checkpoint()

    def step(self, agents):
        self.frame += 1
        for p in self.pipes: p.x -= 2.6
        if self.pipes[-1].x < W - 180: self.pipes.append(Pipe(W + 30))
        audible_bird = max((b for _, b, _ in agents if b.alive), key=lambda b: b.fit, default=None)
        for genome, b, net in agents:
            if not b.alive: continue
            p = next((p for p in self.pipes if p.x + PW >= BX), self.pipes[-1])
            top, bottom = p.mid - GH / 2, p.mid + GH / 2
            out = net.activate((b.y / H, b.v / 10, (p.x - BX) / W, top / H, bottom / H))[0]
            flap = out > .5
            if b is audible_bird and flap:
                self.play_sound("wing", cooldown=.09)
            b.update(flap, self.frame)
            b.fit += .02 + max(0., 1 - abs(b.y - p.mid) / H) * .006
            hit_ground = b.y + 24 > GROUND
            hit_pipe = BX + 28 > p.x and BX < p.x + PW and (b.y < top or b.y + 22 > bottom)
            if b.y < 10 or hit_ground or hit_pipe:
                b.alive = False
                b.fit = max(0., b.fit - 2.)
                if b is audible_bird:
                    self.play_sound("die" if hit_ground else "hit", cooldown=.15)
            score_record = False
            # The network observes its next upcoming pipe, but scoring must
            # check every pipe so the one just passed is not skipped.
            for passed_pipe in self.pipes:
                if passed_pipe.x + PW < BX and passed_pipe.id not in b.passed_pipes:
                    b.passed_pipes.add(passed_pipe.id)
                    b.score += 1
                    b.fit += 12.
                    if b.score > self.best_score:
                        self.best_score = b.score
                        score_record = True
                    if b is audible_bird:
                        self.play_sound("point", cooldown=.12)
            if b.score >= 100 and not b.rewarded_100:
                b.fit += 1000.0
                b.rewarded_100 = True
            genome.fitness = b.fit
            self.gen_fit = max(self.gen_fit, b.fit)
            if b.fit > self.best_fit:
                self.best_fit = b.fit
                self.best_score = max(self.best_score, b.score)
                self.best_genome = genome
                if self.frame % PATH_SAMPLE_EVERY == 0:
                    self.best_path = b.path[:]
                if self.frame % 240 == 0:
                    try:
                        with BEST.open("wb") as f: pickle.dump(genome, f)
                        self.save_data()
                    except OSError: pass
            if score_record:
                self.save_data()
        self.pipes = [p for p in self.pipes if p.x > -PW]
        active_pipe_ids = {p.id for p in self.pipes}
        for _, bird, _ in agents:
            bird.passed_pipes.intersection_update(active_pipe_ids)

    def chat_events(self):
        while self.messages:
            m = self.messages.popleft()
            if m.get("comment", "").strip().lower() == "!changebg":
                self.chatters.add(m.get("user") or "anonymous")
                if len(self.chatters) >= 5:
                    self.fade_from, self.is_night, self.fade_at = self.is_night, not self.is_night, time.monotonic()
                    self.chatters.clear()

    def draw(self, agents):
        self.screen.fill((29, 35, 45))
        gx, gy = 16, 16
        bg = self.assets["night" if self.is_night else "day"]
        if self.fade_at is not None:
            amount = min(1., time.monotonic() - self.fade_at)
            old = self.assets["night" if self.fade_from else "day"]
            self.screen.blit(old, (gx, gy))
            over = bg.copy(); over.set_alpha(round(amount * 255)); self.screen.blit(over, (gx, gy))
            if amount >= 1: self.fade_at = None
        else: self.screen.blit(bg, (gx, gy))
        self.draw_pipes(gx, gy)
        self.ghost(self.previous_path, gx, gy, (65, 225, 255), .60)
        alive = [a for a in agents if a[1].alive]
        if alive:
            current = max(alive, key=lambda a: a[1].fit)[1]
            smoothing = max(.18, .9 / (self.speed ** .5))
            current.visual_y += (current.y - current.visual_y) * smoothing
            self.bird(current.visual_y, current.v, gx, gy, 1.)
        # ground
        pygame.draw.rect(self.screen, (218, 191, 119), (gx, gy + GROUND, W, H - GROUND))
        pygame.draw.rect(self.screen, (111, 189, 68), (gx, gy + GROUND, W, 12))
        # Present the logical game scene in a vertical 9:16 window.
        self.window.fill((17, 20, 26))
        scaled = pygame.transform.smoothscale(self.screen, (WINDOW_SIZE[0], 680))
        self.window.blit(scaled, (0, 0))
        alive = [a for a in agents if a[1].alive]
        score = max((a[1].score for a in alive), default=0)
        generation = getattr(self, "gen", 0)
        stats = (f"Generation: {generation}", f"Highest Score: {self.best_score}", f"Score: {score}")
        for i, text in enumerate(stats):
            rendered = self.font.render(text, True, (245, 245, 245))
            self.window.blit(rendered, rendered.get_rect(center=(WINDOW_SIZE[0] // 2, 705 + i * 31)))
        pygame.display.flip()

    def bird_image(self, vel=0):
        img = self.assets["birds"][(pygame.time.get_ticks() // 110) % 3]
        return pygame.transform.rotate(img, max(-35, min(55, -vel * 4)))

    def bird(self, y, vel, gx, gy, alpha):
        img = self.bird_image(vel).copy()
        if alpha < 1: img.set_alpha(round(255 * alpha))
        self.screen.blit(img, (gx + BX, gy + int(y)))

    def ghost(self, path, gx, gy, color, alpha):
        if not path: return
        position = min(self.frame / PATH_SAMPLE_EVERY, len(path) - 1)
        i = int(position)
        j = min(i + 1, len(path) - 1)
        fraction = position - i
        x = path[i][0] + (path[j][0] - path[i][0]) * fraction
        y = path[i][1] + (path[j][1] - path[i][1]) * fraction
        vel = path[i][2] + (path[j][2] - path[i][2]) * fraction
        img = self.bird_image(vel).copy()
        img.fill((*color, round(alpha * 255)), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(img, (gx + int(x), gy + int(y)))
        points = [(gx + int(px) + 16, gy + int(py) + 12) for px, py, _ in path[max(0, i - 100):i:4]]
        if len(points) > 1: pygame.draw.lines(self.screen, color, False, points, 1)
    def draw_pipes(self, gx, gy):
        smoothing = max(.18, .9 / (self.speed ** .5))
        for p in getattr(self, "pipes", []):
            p.visual_x += (p.x - p.visual_x) * smoothing
            upper = pygame.transform.flip(self.assets["pipe"], False, True)
            x = gx + int(p.visual_x)
            self.screen.blit(upper, (x, gy + int(p.mid - GH/2) - 320))
            self.screen.blit(self.assets["pipe"], (x, gy + int(p.mid + GH/2)))

    def run(self):
        self.pop = self.population()
        self.restart = False
        while self.running:
            self.pop.run(self.evaluate, 1)
            if self.restart:
                self.pop = self.new_population()
                self.restart = False
        self.checkpoint(); self.chat.stop(); pygame.quit()


if __name__ == "__main__":
    try: Game().run()
    except KeyboardInterrupt: pygame.quit(); sys.exit(0)
